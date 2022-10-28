import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function2, BaseNetwork
from model.networks.base_network import freeze_network, init_network, print_network
import model.networks as network
# import model.networks.dior_models as dior_models
# import model.networks.dior_multi_model as dior_multi_model
import model.networks.dior_single_model2 as dior_single_model
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os


class Dior2(BaseModel):
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
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_cx', type=float, default=0.1, help='weight for the VGG19 context loss')

        parser.add_argument('--lambda_gb', type=float, default=2.0, help='weight for adv generation on bones loss')
        parser.add_argument('--lambda_gs', type=float, default=2.0, help='weight for adv generation on segment loss')
        parser.add_argument('--lambda_gi', type=float, default=2.0, help='weight for adv generation on image loss')
        
        parser.add_argument('--lambda_seg', type=float, default=1, help='weight soft mask segmentation')

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
        # self.loss_names = ['correctness_gen', 'regularization_flow', 'l1', 'content_gen', 'style_gen', 'cx_gen', 'mask',
        #                     'g_pb', 'g_pp', 'g_ps', 'g_pb_features', 'g_pp_features', 'g_ps_features',
        #                     'd_pb', 'd_pp', 'd_ps']
        self.loss_names = ['l1', 'content_gen', 'style_gen', 'cx_gen', 'mask',
                            'g_pb', 'g_pp', 'g_ps', 'g_pb_features', 'g_pp_features', 'g_ps_features',
                            'd_pb', 'd_pp', 'd_ps']
        # self.loss_names = ['l1', 'content_gen', 'style_gen', 'mask',
        #                     'g_pb', 'g_pp', 'g_ps',
        #                     'd_pb', 'd_pp', 'd_ps']

        # self.visual_names = ['input_P1','input_P2','input_BP1', 'input_BP2','input_SP1','input_SP2',
        #     'img_gen', 'flow_fields','soft_mask_list','inter_images'
        #     ]
        self.visual_names = ['input_P1','input_P2','input_BP1', 'input_BP2','input_SP1','input_SP2','input_M1',
            'img_gen', 'soft_mask_list','inter_images'
            ]
        self.model_names = ['Enc','Dec', 'DiorG','D_PB','D_PP','D_PS']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        
        self.vgg = external_function2.VGG19Limited('./saved_models/vgg19-dcbb9e9d.pth').to(opt.device)

        self.net_Enc = init_network(
            dior_single_model.Encoder2(input_img_dim=3,style_dim=256, norm='none',activation='LeakyReLU',pad_type='reflect')
            , opt,init_type='kaiming')
        self.net_Dec = init_network(
            dior_single_model.Decoder2(input_dim=256, output_dim=3,norm='layer',n_res = 2)
            , opt,init_type='kaiming')

        self.net_FlowG = init_network(network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18, ngf=64, img_f=512,
                        encoder_layer=4, attn_layer=opt.attn_layer,
                        norm='instance', activation='LeakyReLU',
                        use_spect=opt.use_spect_g, use_coord=False), opt)

        
        self.net_DiorG = init_network(dior_single_model.Generator2(image_nc=3, structure_nc=18, 
            norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False), opt)
        
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
                    input_nc=3+18, ndf=64, img_f=128, layers=3, n_res=2, multi_dim_out=True,
                    norm='instance', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)
            # segment conditioned
            if opt.with_D_PS:
                from data.dior2_dataset import SEG
                # self.net_D_PS=init_network(network.discriminator.ResDiscriminator(
                #     input_nc=3+len(SEG.labels), ndf=64, img_f=128, layers=4, norm='instance', 
                #     activation='LeakyReLU', use_spect=opt.use_spect_d,
                #     use_coord=False), opt)
                self.net_D_PS=init_network(external_function2.PatchDiscriminatorWithLayers(
                    input_nc=3+len(SEG.labels), ndf=64, img_f=128, layers=3, n_res=2, multi_dim_out=True,
                    norm='instance', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)
            # photo conditioned
            if opt.with_D_PP:
                # self.net_D_PP=init_network(network.discriminator.ResDiscriminator(
                #     input_nc=3+3, ndf=64, img_f=128, layers=4, norm='instance', activation='LeakyReLU', 
                #     use_spect=opt.use_spect_d,
                #     use_coord=False), opt)
                self.net_D_PP=init_network(external_function2.PatchDiscriminatorWithLayers(
                    input_nc=3+3, ndf=64, img_f=128, layers=3, n_res=2, multi_dim_out=True,
                    norm='instance', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False), opt)

        if self.isTrain:
            # define the loss functions
            # geo loss
            self.Correctness = external_function2.PerceptualCorrectness().to(opt.device)
            # self.Correctness.vgg = self.vgg
            self.Regularization = external_function2.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)

            # content loss:l1 + content + style
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function2.VGGLoss().to(opt.device)
            # self.Vggloss.vgg = self.vgg
            self.Cxloss = external_function2.ContextSimilarityLoss(sigma=0.5).to(opt.device)
            # self.Cxloss.vgg =self.vgg

            # self.PerceptualLoss = external_function2.PerceptualLoss().to(opt.device)
            # self.PerceptualLoss.vgg = self.vgg
            # gan loss
            self.GANloss = external_function2.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.GanFeaturesLoss = external_function2.FeatureMappingsLoss()
            # segmentation loss
            self.SegSoftmaskloss = torch.nn.BCELoss()       
           

            # freeze flow model
            base_function._freeze(self.net_FlowG)
            # base_function._freeze(self.net_Enc)
            # base_function._unfreeze(self.net_Enc.mask2)
            # base_function._freeze(self.net_Dec)
            # define the optimizer
            self.optimizer_G = torch.optim.AdamW(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_DiorG.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_Dec.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_Enc.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_FlowG.parameters())),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizer_DPB = torch.optim.AdamW(itertools.chain(
                                    filter(lambda p: p.requires_grad, self.net_D_PB.parameters())),
                                    lr=opt.lr*opt.ratio_g2d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DPB)
            if opt.with_D_PS:
                self.optimizer_DPS = torch.optim.AdamW(itertools.chain(
                                    filter(lambda p: p.requires_grad, self.net_D_PS.parameters())),
                                    lr=opt.lr*opt.ratio_g2d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DPS)
            if opt.with_D_PP:
                self.optimizer_DPP = torch.optim.AdamW(itertools.chain(
                                    filter(lambda p: p.requires_grad, self.net_D_PP.parameters())),
                                    lr=opt.lr*opt.ratio_g2d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_DPP)
            
        # load the pre-trained model and schedulers
        self.setup(opt)

        #custom schedulers
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2")
        # self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        print('---------- Networks initialized -------------')
        print_network(self.net_FlowG)
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
        # self.input_P1 = self.input_P1 * self.input_M1.unsqueeze(1)
        self.input_SP1 = self.input_SP1 * self.input_M1.unsqueeze(1)
        """Run forward processing to get the inputs"""
        # self.flow_fields = self.net_FlowG(self.input_P1, self.input_BP1, self.input_BP2)[0]
        # take h/4, w/4 flow                
        # flow = self.flow_fields[-1]

        #noflow pretrain
        # b,_,h,w = self.input_P1.size()
        # flow = torch.zeros((b,2,int(h/4),int(w/4)), device=self.input_P1.device)
        flow=None

        self.img_gen,self.soft_mask_list, self.inter_images = self.net_DiorG(self.vgg,
            self.net_Enc,self.net_Dec, flow,
            self.input_P1, self.input_BP1, self.input_SP1, self.input_BP2, self.input_SP2)


    def backward_G(self):
        """Calculate training loss for the generator"""
        # vgg features
        vgg_features_img_gen =self.vgg(self.img_gen)
        vgg_features_p1, vgg_features_p2  = self.vgg(self.input_P1), self.vgg(self.input_P2) 

        # # calculate geometric loss #####
        # # Calculate Sampling Correctness Loss
        # # loss_correctness_gen = self.Correctness(self.input_P2, self.input_P1, self.flow_fields, self.opt.attn_layer)
        # loss_correctness_gen = self.Correctness(vgg_features_p2, vgg_features_p1, self.flow_fields, self.opt.attn_layer)
        # self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct

        # # Calculate regularization term
        # loss_regularization_flow = self.Regularization(self.flow_fields)
        # self.loss_regularization_flow = loss_regularization_flow * self.opt.lambda_regularization

        ########## Content loss #######################
        # Calculate l1 loss
        loss_l1 = self.L1loss(self.img_gen, self.input_P2)
        self.loss_l1 = loss_l1 * self.opt.lambda_rec

        # Calculate perceptual loss
        # loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2)
        loss_content_gen, loss_style_gen = self.Vggloss(vgg_features_img_gen, vgg_features_p2)
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content

        #calculate contextual loss
        # cx_loss = self.Cxloss(self.img_gen, self.input_P2)
        cx_loss = self.Cxloss(vgg_features_img_gen, vgg_features_p2)
        self.loss_cx_gen = cx_loss *self.opt.lambda_cx

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

        ########## Segmentation mask loss #################
        loss_mask = 0
        for soft_mask_target_mask in self.soft_mask_list:
            soft_mask, target_mask = soft_mask_target_mask
            loss_mask+=self.SegSoftmaskloss(soft_mask,target_mask)
        # loss_mask = loss_mask / len(self.soft_mask_list)
        self.loss_mask = loss_mask * self.opt.lambda_seg      

        # ######### Intermediate image loss #########
        # intermediate_image_loss = 0
        # skin_mask_index = 2
        # for index, inter_image in enumerate(self.inter_images):
        #     soft_mask, target_mask = self.soft_mask_list[index+skin_mask_index]
        #     i_pred = nn.Upsample(scale_factor=4, mode='bilinear')(soft_mask) * inter_image
        #     i_real = nn.Upsample(scale_factor=4, mode='bilinear')(target_mask) * self.input_P2
        #     # import cv2;cv2.imwrite('obody.png',12 * nn.Upsample(scale_factor=4, mode='bilinear')(soft_mask).detach().cpu().numpy())
        #     # import cv2;cv2.imwrite('obody.png',127 + 128 * torch.cat((i_real[0],i_pred[0]),-1).detach().permute(1,2,0).cpu().numpy())
        #     inter_l1_loss = torch.nn.MSELoss()(i_pred,i_real)
        #     perc_loss = self.PerceptualLoss(i_pred,i_real)
        #     intermediate_image_loss += inter_l1_loss +perc_loss
        #     # inter_loss_content_gen, inter_loss_style_gen = self.Vggloss(i_pred,i_real)
        #     # intermediate_image_loss += inter_l1_loss+ (inter_loss_content_gen + inter_loss_style_gen)
        # self.loss_inter = intermediate_image_loss * 0.1

        
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

        
