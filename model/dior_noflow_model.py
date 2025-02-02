import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function2, BaseNetwork
from model.networks.base_network import freeze_network, init_network, print_network
import model.networks as network
# import model.networks.dior_models as dior_models
# import model.networks.dior_multi_model as dior_multi_model
import model.networks.dior_single_model as dior_single_model
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os


class Dior_noflow(BaseModel):
    """
       Deep Spatial Transformation For Dior Based Image Generation
    """
    def name(self):
        return "Dress in order Person Image Generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--style_dim', type=int, default=128, help='dimension of style vector')
        parser.add_argument('--style_blocks', type=int, default=7, help='number of style blocks')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1 parameter')

        
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for the affine regularization loss')

        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')

        parser.add_argument('--lambda_gb', type=float, default=2.0, help='weight for adv generation on bones loss')
        parser.add_argument('--lambda_gs', type=float, default=2.0, help='weight for adv generation on segment loss')
        parser.add_argument('--lambda_gi', type=float, default=2.0, help='weight for adv generation on image loss')
        
        parser.add_argument('--lambda_seg', type=float, default=0.1, help='weight soft mask segmentation')

        parser.add_argument('--with_D_PP', type=int, default=1, help='use D to judge P and P is pair or not')
        parser.add_argument('--with_D_PB', type=int, default=1, help='use D to judge P and B is pair or not')
        parser.add_argument('--with_D_PS', type=int, default=1, help='use D to judge P and S is pair or not')
        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        # parser.set_defaults(use_spect_d=False)
        parser.set_defaults(save_input=False)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['l1', 'content_gen', 'style_gen', 'mask',
                            'g_pb', 'g_pp', 'g_ps',
                            'd_pb', 'd_pp', 'd_ps']
        # self.loss_names = ['l1', 'content_gen', 'style_gen', 'mask']

        self.visual_names = ['input_P1','input_P2','input_BP1', 'input_BP2','input_SP1','input_SP2',
            'img_gen', 'soft_mask_list','inter_images'
            ]
        self.model_names = ['Enc','Dec', 'DiorG','D_PB','D_PP','D_PS']
        # self.model_names = ['Enc','Dec', 'DiorG']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.vgg = external_function2.VGG19Limited('./saved_models/vgg19-dcbb9e9d.pth').to(opt.device)
        self.net_Enc = init_network(dior_single_model.Encoder2(input_img_dim=3,style_dim=256, norm='none',activation='LeakyReLU',pad_type='reflect'), opt,init_type='kaiming')
        self.net_Enc.vgg = self.vgg
        self.net_Dec = init_network(dior_single_model.Decoder2(input_dim=256, output_dim=3,norm='layer'), opt,init_type='kaiming')
        
        self.net_DiorG = init_network(dior_single_model.Generator2(image_nc=3, structure_nc=18, 
            norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False), opt)
    
        if self.isTrain:
            # TODO check network normalization
            # pose conditioned
            if opt.with_D_PB:                
                self.net_D_PB=init_network(network.discriminator.ResDiscriminator(
                    input_nc=3+18, ndf=64, img_f=128, layers=4, norm='instance',
                    activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)
            # segment conditioned
            if opt.with_D_PS:
                from data.dior2_dataset import SEG
                self.net_D_PS=init_network(network.discriminator.ResDiscriminator(
                    input_nc=3+len(SEG.labels), ndf=64, img_f=128, layers=4, norm='instance', 
                    activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)
            # photo conditioned
            if opt.with_D_PP:
                self.net_D_PP=init_network(network.discriminator.ResDiscriminator(
                    input_nc=3+3, ndf=64, img_f=128, layers=4, norm='instance', activation='LeakyReLU', 
                    use_spect=opt.use_spect_d,
                    use_coord=False), opt)

        if self.isTrain:
            # define the loss functions
            # content loss:l1 + content + style
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function2.VGGLoss().to(opt.device)
            self.Vggloss.vgg = self.vgg

            # gan loss
            self.GANloss = external_function2.AdversarialLoss(opt.gan_mode).to(opt.device)
            
            # segmentation loss
            self.SegSoftmaskloss = torch.nn.BCELoss()       
           

            # freeze flow model
            # base_function._freeze(self.net_FlowG)
            # base_function._freeze(self.net_Enc)
            # base_function._unfreeze(self.net_Enc.mask2)
            # base_function._freeze(self.net_Dec)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_DiorG.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_Dec.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_Enc.parameters())),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizer_DPB = torch.optim.Adam(itertools.chain(
                                    filter(lambda p: p.requires_grad, self.net_D_PB.parameters())),
                                    lr=opt.lr*opt.ratio_g2d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DPB)
            if opt.with_D_PS:
                self.optimizer_DPS = torch.optim.Adam(itertools.chain(
                                    filter(lambda p: p.requires_grad, self.net_D_PS.parameters())),
                                    lr=opt.lr*opt.ratio_g2d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DPS)
            if opt.with_D_PP:
                self.optimizer_DPP = torch.optim.Adam(itertools.chain(
                                    filter(lambda p: p.requires_grad, self.net_D_PP.parameters())),
                                    lr=opt.lr*opt.ratio_g2d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DPP)
            
        # load the pre-trained model and schedulers
        self.setup(opt)

        print('---------- Networks initialized -------------')
        print_network(self.net_Dec)
        print_network(self.net_Enc)
        print_network(self.net_DiorG)
        if self.isTrain:
            if opt.with_D_PB:
                print_network(self.net_D_PB)
            if opt.with_D_PP:
                print_network(self.net_D_PP)
            if opt.with_D_PS:
                print_network(self.net_D_PS)
        print('-----------------------------------------------')
    

    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1, input_BP1, input_SP1, input_M1 = input['P1'], input['BP1'], input['SP1'], input['M1']
        input_P2, input_BP2, input_SP2 = input['P2'], input['BP2'], input['SP2']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_SP1 = input_SP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_SP2 = input_SP2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_M1 = input_M1.cuda(self.gpu_ids[0], non_blocking=True)

        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])


    def forward(self):
        self.input_P1 = self.input_P1 * self.input_M1.unsqueeze(1)
        self.input_SP1 = self.input_SP1 * self.input_M1.unsqueeze(1)
        """Run forward processing to get the inputs"""
        b,_,h,w = self.input_P1.size()
        flow = torch.zeros((b,2,int(h/4),int(w/4)), device=self.input_P1.device)
        self.img_gen,self.soft_mask_list, self.inter_images = self.net_DiorG(
            self.net_Enc,self.net_Dec, flow,
            self.input_P1, self.input_BP1, self.input_SP1, self.input_BP2, self.input_SP2)


    def backward_G(self):
        """Calculate training loss for the generator"""
        # calculate geometric loss #####
        ########## Content loss #######################
        # Calculate l1 loss
        loss_l1 = self.L1loss(self.img_gen, self.input_P2)
        self.loss_l1 = loss_l1 * self.opt.lambda_rec

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2)  
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content

        ########## Gan loss #######################

        # Calculate GAN loss
        if self.opt.with_D_PB:
            base_function._freeze(self.net_D_PB)
            D_fake = self.net_D_PB(torch.cat((self.img_gen, self.input_BP2),1))
            self.loss_g_pb = self.GANloss(D_fake, True, False) * self.opt.lambda_gb
        if self.opt.with_D_PS:
            base_function._freeze(self.net_D_PS)
            D_fake = self.net_D_PS(torch.cat((self.img_gen, self.input_SP2),1))
            self.loss_g_ps = self.GANloss(D_fake, True, False) * self.opt.lambda_gs
        if self.opt.with_D_PP:
            base_function._freeze(self.net_D_PP)
            D_fake = self.net_D_PP(torch.cat((self.img_gen, self.input_P1),1))
            self.loss_g_pp = self.GANloss(D_fake, True, False) * self.opt.lambda_gi

        ########## Segmentation mask loss #################
        loss_mask = 0
        for soft_mask_target_mask in self.soft_mask_list:
            soft_mask, target_mask = soft_mask_target_mask
            loss_mask+=self.SegSoftmaskloss(soft_mask,target_mask)
        # loss_mask = loss_mask / len(self.soft_mask_list)
        self.loss_mask = loss_mask * self.opt.lambda_seg        
        
        # total weighted loss
        total_gen_loss = 0
        for name in self.loss_names:
            if not name.startswith('d_') :# and hasattr(self, "loss_" + name)
                total_gen_loss += getattr(self, "loss_" + name)
        total_gen_loss.backward()
        pass


    def backward_D_basic(self,  optimizer, netD, real, fake):
            optimizer.zero_grad()
            base_function._unfreeze(netD)

            """Calculate GAN loss for the discriminator"""
            # Real
            D_real = netD(real)
            D_real_loss = self.GANloss(D_real, True, True)
            # fake
            D_fake = netD(fake.detach())
            D_fake_loss = self.GANloss(D_fake, False, True)
            # loss for discriminator
            D_loss = (D_real_loss + D_fake_loss) * 0.5
            # gradient penalty for wgan-gp
            if self.opt.gan_mode == 'wgangp':
                gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
                D_loss += gradient_penalty

            D_loss.backward()

            optimizer.step()
            return D_loss



    def optimize_parameters(self):
        """update network weights"""
        # with torch.cuda.amp.autocast(enabled=True):
        self.forward()

        if self.opt.with_D_PB:
            self.loss_d_pb = self.backward_D_basic(
                self.optimizer_DPB,
                self.net_D_PB,
                real=torch.cat((self.input_P2, self.input_BP2),1),
                fake=torch.cat((self.img_gen, self.input_BP2),1)
            )
        if self.opt.with_D_PP:
            self.loss_d_pp = self.backward_D_basic(
                self.optimizer_DPP,
                self.net_D_PP,
                real=torch.cat((self.input_P2, self.input_P1),1),
                fake=torch.cat((self.img_gen, self.input_P1),1)
            )
        if self.opt.with_D_PS:
            self.loss_d_ps = self.backward_D_basic(
                self.optimizer_DPS,
                self.net_D_PS,
                real=torch.cat((self.input_P2, self.input_SP2),1),
                fake=torch.cat((self.img_gen, self.input_SP2),1)
            )

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        
