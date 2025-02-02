import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function2, BaseNetwork
from model.networks.base_network import freeze_network, init_network, print_network
import model.networks as network
# import model.networks.dior_models as dior_models
# import model.networks.dior_multi_model as dior_multi_model
import model.networks.sawn_model as sawn_model
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os


class Sawn(BaseModel):
    """
       Deep Spatial Transformation For Dior Based Image Generation
    """
    def name(self):
        return "Dress in order Person Image Generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...", help="The number layers away from output layer")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")
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
        parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.05, help='weight for the VGG19 content loss')

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
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['correctness_gen', 'regularization_flow', 'l1', 'content_gen', 'style_gen',
                            'g_pb', 'g_pp', 'g_ps', 'g_pb_features', 'g_pp_features', 'g_ps_features',
                            'd_pb', 'd_pp', 'd_ps']
        # self.loss_names = ['l1', 'content_gen', 'style_gen', 'mask',
        #                     'g_pb', 'g_pp', 'g_ps',
        #                     'd_pb', 'd_pp', 'd_ps']

        self.visual_names = ['input_P1','input_P2','input_BP1', 'input_BP2','input_SP1','input_SP2',
            'img_gen', 'flow_fields','flow_masks'
            ]
        self.model_names = ['FlowG','Enc','Gen', 'D_PB','D_PP','D_PS']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        
        self.vgg = external_function2.VGG19Limited('./saved_models/vgg19-dcbb9e9d.pth').to(opt.device)

        

        from data.dior2_dataset import SEG
        self.net_Enc = init_network(sawn_model.ImageEncoder(
                input_img_dim=3,n_layers=3,k_classes=len(SEG.labels),norm='none'
            ), opt,init_type='kaiming')
        
        self.net_Gen = init_network(sawn_model.Generator(
                n_layers=3
            ), opt,init_type='orthogonal')

        self.net_FlowG = init_network(network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18, ngf=64, img_f=512,
                        encoder_layer=4, attn_layer=opt.attn_layer,
                        norm='instance', activation='LeakyReLU',
                        use_spect=opt.use_spect_g, use_coord=False), opt)
        
        
        
        self.flow2color = util.flow2color()
    
        if self.isTrain:
            # TODO check network normalization
            # pose conditioned
            if opt.with_D_PB:                
                # self.net_D_PB=init_network(network.discriminator.ResDiscriminator(
                #     input_nc=3+18, ndf=64, img_f=128, layers=4, norm='instance',
                #     activation='LeakyReLU', use_spect=opt.use_spect_d,
                #     use_coord=False), opt)
                self.net_D_PB=init_network(external_function2.PatchDiscriminatorWithLayers(
                    input_nc=3+18, ndf=64, img_f=256, layers=4, norm='instance', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)
            # segment conditioned
            if opt.with_D_PS:
                
                # self.net_D_PS=init_network(network.discriminator.ResDiscriminator(
                #     input_nc=3+len(SEG.labels), ndf=64, img_f=128, layers=4, norm='instance', 
                #     activation='LeakyReLU', use_spect=opt.use_spect_d,
                #     use_coord=False), opt)
                self.net_D_PS=init_network(external_function2.PatchDiscriminatorWithLayers(
                    input_nc=3+len(SEG.labels), ndf=64, img_f=256, layers=4, norm='instance', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)
            # photo conditioned
            if opt.with_D_PP:
                # self.net_D_PP=init_network(network.discriminator.ResDiscriminator(
                #     input_nc=3+3, ndf=64, img_f=128, layers=4, norm='instance', activation='LeakyReLU', 
                #     use_spect=opt.use_spect_d,
                #     use_coord=False), opt)
                self.net_D_PP=init_network(external_function2.PatchDiscriminatorWithLayers(
                    input_nc=3+3, ndf=64, img_f=256, layers=4, norm='instance', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)

        if self.isTrain:
            # define the loss functions
            # geo flow loss
            self.Correctness = external_function2.PerceptualCorrectness().to(opt.device)
            self.Correctness.vgg = self.vgg
            self.Regularization = external_function2.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)

            # content loss:l1 + content + style
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function2.VGGLoss().to(opt.device)
            self.Vggloss.vgg = self.vgg

            # self.PerceptualLoss = external_function2.PerceptualLoss().to(opt.device)
            # self.PerceptualLoss.vgg = self.vgg
            # gan loss
            self.GANloss = external_function2.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.GanFeaturesLoss = external_function2.FeatureMappingsLoss()
            
            # freeze flow model
            base_function._freeze(self.net_FlowG)
            base_function._unfreeze(self.net_FlowG.flow_net.mask0)
            base_function._unfreeze(self.net_FlowG.flow_net.mask1)
            base_function._unfreeze(self.net_FlowG.flow_net.mask2)
            # base_function._freeze(self.net_Enc)
            # base_function._unfreeze(self.net_Enc.mask2)
            # base_function._freeze(self.net_Dec)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_Gen.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_Enc.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_FlowG.parameters())),
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
        print_network(self.net_FlowG)
        print_network(self.net_Enc)
        print_network(self.net_Gen)
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
        # self.flow_fields,self.flow_masks = self.net_FlowG(self.input_P1, self.input_BP1, self.input_BP2)
        # layers, gammas, betas = self.net_Enc(self.input_P1, self.input_SP1, self.vgg)
        self.flow_fields, self.flow_masks, self.img_gen = self.net_Gen(self.input_P1, self.input_BP1, self.input_SP1, self.input_BP2,
            self.net_FlowG, self.net_Enc, self.vgg)
        pass


    def backward_G(self):
        """Calculate training loss for the generator"""
        # calculate geometric loss #####
        # Calculate Sampling Correctness Loss
        loss_correctness_gen = self.Correctness(self.input_P2, self.input_P1, self.flow_fields, self.opt.attn_layer)
        self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct

        # Calculate regularization term
        loss_regularization_flow = self.Regularization(self.flow_fields)
        self.loss_regularization_flow = loss_regularization_flow * self.opt.lambda_regularization

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
            D_fake, features_fake = self.net_D_PB(torch.cat((self.img_gen, self.input_BP2),1))
            D_real, features_real = self.net_D_PB(torch.cat((self.input_P2, self.input_BP2),1))
            gan_loss = self.GANloss(D_fake, True, False)
            self.loss_g_pb = gan_loss * self.opt.lambda_gb
            self.loss_g_pb_features = torch.tensor(0)
            if gan_loss > 0.5:
                self.loss_g_pb_features = self.GanFeaturesLoss(features_fake,features_real) * max(gan_loss.item()-0.5,0)
        if self.opt.with_D_PS:
            base_function._freeze(self.net_D_PS)
            D_fake, features_fake = self.net_D_PS(torch.cat((self.img_gen, self.input_SP2),1))
            D_real, features_real = self.net_D_PS(torch.cat((self.input_P2, self.input_SP2),1))
            gan_loss = self.GANloss(D_fake, True, False)
            self.loss_g_ps = gan_loss * self.opt.lambda_gs
            self.loss_g_ps_features = torch.tensor(0)
            if gan_loss > 0.5:
                self.loss_g_ps_features = self.GanFeaturesLoss(features_fake,features_real) * max(gan_loss.item()-0.5,0)
        if self.opt.with_D_PP:
            base_function._freeze(self.net_D_PP)
            D_fake, features_fake = self.net_D_PP(torch.cat((self.img_gen, self.input_P1),1))
            D_real, features_real = self.net_D_PP(torch.cat((self.input_P2, self.input_P1),1))
            gan_loss = self.GANloss(D_fake, True, False)
            self.loss_g_pp = gan_loss * self.opt.lambda_gi
            self.loss_g_pp_features = torch.tensor(0)
            if gan_loss > 0.5:
                self.loss_g_pp_features = self.GanFeaturesLoss(features_fake,features_real) * max(gan_loss.item()-0.5,0)

        
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
            D_real, _ = netD(real)
            D_real_loss = self.GANloss(D_real, True, True)
            # fake
            D_fake, _ = netD(fake.detach())
            D_fake_loss = self.GANloss(D_fake, False, True)
            # loss for discriminator
            D_loss = (D_real_loss + D_fake_loss) * 0.5
            # gradient penalty for wgan-gp
            # if self.opt.gan_mode == 'wgangp':
            #     gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            #     D_loss += gradient_penalty
            

            D_loss.backward()

            optimizer.step()
            return D_loss



    def optimize_parameters(self):
        """update network weights"""
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

        
