import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function, BaseNetwork
from data.dior_dataset import SEG
import model.networks as network
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os

class Conv2dBlock(BaseNetwork):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='ReLU', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.pad = base_function.get_padding(pad_type, padding)()
        kwargs_conv = {'kernel_size': kernel_size, 'stride': stride}
        self.conv = base_function.coord_conv(input_dim, output_dim, **kwargs_conv)
        self.activation = base_function.get_nonlinearity_layer(activation)
        if norm != 'none':
            self.norm = base_function.get_norm_layer(norm)(output_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if hasattr(self,'norm'):
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# class SoftShapeMask(BaseNetwork):
#     def __init__(self, input_dim, norm, num_blocks=2):
#         super(SoftShapeMask, self).__init__()
#         self.model = [base_function.ResBlocks(num_blocks=num_blocks, input_nc = input_dim, hidden_nc=64,
#             output_nc=1, norm_layer = base_function.get_norm_layer(norm))]
#         self.model += [nn.Sigmoid()]
#         self.model = nn.Sequential(*self.model)

#     def forward(self, x):
#         return self.model(x)


class SegmentEncoder(BaseNetwork):
    def __init__(self,n_downsample, input_dim, dim, style_dim, norm='instance', activation='LeakyReLU', pad_type ='reflect'):
        super(SegmentEncoder, self).__init__()
        self.add_module('vgg', external_function.VGG19())
        self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm,activation=activation, pad_type=pad_type)# 3->64,concat
        dim = dim * 2
        self.conv2 = Conv2dBlock(dim, dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 128->128,concat
        dim = dim * 2
        self.conv3 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 256->256,concat
        dim = dim * 2
        # self.conv4 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 512->512,concat
        # dim = dim * 2
        self.enc = nn.Sequential(*[nn.Conv2d(dim, style_dim, 1, 1, 0)])
        # self.bilinear_warp = base_function.BilinearSamplingBlock()
        # self.soft_mask = SoftShapeMask(style_dim,norm,num_blocks=2)
        self.soft_mask = nn.Sequential(*[base_function.ResBlock(128,1,norm_layer=None),nn.Sigmoid()])


    def forward(self, seg_img, sem_mask, flow):
        xi = seg_img * sem_mask.repeat(1, seg_img.size(1), 1, 1)
        # import cv2;cv2.imwrite('oo2.png',127 + 127 * torch.cat((xi,util.bilinear_warp(xi, flow)),-1)[0].permute(1,2,0).cpu().detach().numpy())
        sty_fea = self.vgg(xi)
        x = self.conv1(xi)
        x = torch.cat([x, sty_fea['relu1_1']], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, sty_fea['relu2_1']], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, sty_fea['relu3_1']], dim=1)
        # x = self.conv4(x)
        # x = torch.cat([x, sty_fea['relu4_1']], dim=1)
        x = self.enc(x)
        # warp by flow field using bilinear interpolation
        tx = util.bilinear_warp(x, flow)
        # soft mask torch.nn.BCELoss()(mx[0][0],torch.nn.functional.interpolate(sem_mask.type(torch.float),size=mx.shape[-2:])[0][0])
        # nn.Sequential(*[nn.Conv2d(128,64,3,1,1), nn.LeakyReLU(),nn.Conv2d(64,1,3,1,1),nn.Sigmoid()])
        # import cv2;cv2.imwrite('oo1.png',254 * sem_mask.type(torch.float)[0][0].cpu().numpy())
        # import cv2;cv2.imwrite('oo2s.png',254 * torch.cat((util.bilinear_warp(sem_mask.type(torch.float), flow),F.interpolate(mx,sem_mask.shape[2:])),-1)[0][0].detach().cpu().numpy())
        mx = self.soft_mask(tx)

        # segment loss and view
        # import cv2;cv2.imwrite('oo_s.png',254 * torch.cat((seg_small,_mask),-1)[0][0].cpu().detach().numpy())
        # warped_seg = util.bilinear_warp(sem_mask.type(torch.float),flow)
        # seg_small = F.interpolate(warped_seg,mx.shape[-2:]).detach()
        # soft_mask_list.append(torch.cat((mx,seg_small),-1))
        return tx, mx


class PoseEncoder(BaseNetwork):
    def __init__(self,n_downsample, input_dim, dim, max_dim = 256, n_res = 2, norm='instance', 
                 activation='LeakyReLU', pad_type = 'zero'):
        super(PoseEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            next_dim = min( max_dim, dim * 2)
            self.model += [Conv2dBlock(dim, next_dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
            dim = min( max_dim, dim * 2)
        #  residual blocks
        if n_res>0:
            self.model += [base_function.ResBlocks(num_blocks=n_res, input_nc = dim, 
                norm_layer = base_function.get_norm_layer(norm))]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class StyleGenerator(BaseNetwork):
    def __init__(self,n_blocks,dim):
        super(StyleGenerator, self).__init__()
        # input_nc, output_nc, hidden_nc, label_nc, spade_config_str='spadeinstance3x3', nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False, learned_shortcut=False
        self.blocks = []
        for i in range(n_blocks):
            self.blocks += [base_function.SPADEResnetBlock(input_nc = dim, output_nc = dim,
                                                          hidden_nc = dim, label_nc =dim,
                                                          spade_config_str='spadeinstance3x3')]
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, prev_state, texture):
        state = prev_state
        for i, block in enumerate(self.blocks):
            state = block(state, texture)
        return state


class Decoder(BaseNetwork):
    def __init__(self,n_upsample,input_dim, output_dim, norm = 'layer', activation = nn.LeakyReLU):
        super(Decoder, self).__init__()
        self.model = []
        self.up = nn.Upsample(scale_factor=2)
        # self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d
        for i in range(n_upsample):
            # self.model += [base_function.ResBlockDecoder(input_dim, input_dim//2,
            #     norm_layer = base_function.get_norm_layer(norm))]
            self.model += [self.up]
            self.model += [base_function.ResBlock(input_dim, input_dim//2,
                learnable_shortcut=True, norm_layer=base_function.get_norm_layer(norm))]
            input_dim = input_dim // 2
        self.model += [Conv2dBlock(input_dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='zero')]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)


class DiorGenerator(BaseNetwork):
    def __init__(self,opt):
        super(DiorGenerator, self).__init__()
        # flow field generator
        # self.flow_net =network.generator.PoseFlowNet(image_nc=3, structure_nc=18, ngf=32, img_f=256,
        #                 encoder_layer=5, attn_layer=[1,2],
        #                 norm='instance', activation='LeakyReLU',
        #                 use_spect=opt.use_spect_g, use_coord=False)
        self.pose_encoder= init_network(PoseEncoder(n_downsample=2,
            input_dim=18,dim=64, max_dim=opt.style_dim, n_res=2,norm='instance'),opt,init_type='xavier')

        self.flow_net =init_network(network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18, ngf=32, img_f=128,
                        encoder_layer=6, attn_layer=[1,2],
                        norm='instance', activation='LeakyReLU',
                        use_spect=opt.use_spect_g, use_coord=False), opt)
        self.segment_encoder = init_network(SegmentEncoder(n_downsample=2,
            input_dim=3,dim=64, style_dim=opt.style_dim, norm='instance'),
            opt,init_type='xavier')
        
        self.decoder = init_network(Decoder(n_upsample=2,input_dim=opt.style_dim,output_dim=3,norm='layer'),opt,init_type='xavier')
        self.style_gen = init_network(StyleGenerator(n_blocks=opt.style_blocks, dim=opt.style_dim),opt)


    def recur_wear(self, img, source_Seg, target_Seg, seg_id, flow, prev_style):
        seg = source_Seg[:,seg_id,...].unsqueeze(1)
        t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
        _texture, _mask = self.segment_encoder(img, seg, flow)
        self.soft_mask_difference(_mask,t_seg)
        # warped_seg = util.bilinear_warp(seg.type(torch.float),flow)
        # seg_small = F.interpolate(warped_seg,_mask.shape[-2:]).detach()
        # warp seg and interpolate import cv2;cv2.imwrite('oo_s.png',254 * torch.cat((seg_small,_mask),-1)[0][0].cpu().detach().numpy())
        # self.masks_validation[seg_id]=(_mask,seg_small)
        z_style_next = self.style_gen(prev_style, _texture)
        z_style_next = z_style_next * _mask + prev_style * (1 - _mask)
        # self.soft_mask_list.append(torch.cat((_mask,seg_small),-1))
        
        return z_style_next


    # def flow_estimate(self,source_Img, source_Pose, target_Pose):
    #     flow_fields = self.flow_net(source_Img, source_Pose, target_Pose)
    #     return flow_fields[0][0]
    
    def soft_mask_difference(self, mask, seg):
        self.soft_mask_list.append([mask,F.interpolate(seg.float(),mask.shape[2:])])
        pass

    def _has_seg_values(seg):
        pass


    def forward(self, source_Img, source_Pose, source_Seg, source_Mask, target_Pose, target_Seg):
        flow_fields = self.flow_net(source_Img, source_Pose, target_Pose)[0]
        # take h/4, w/4 flow        
        flow = flow_fields[0]
        # masking for inpainting        
        source_Img = source_Img * source_Mask.unsqueeze(1)
        source_Seg = source_Seg * source_Mask.unsqueeze(1)

        bg_Seg = source_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        target_bg_mask = target_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        face_seg = source_Seg[:,SEG.FACE,...].unsqueeze(1)
        target_face_mask = target_Seg[:,SEG.FACE,...].unsqueeze(1)
        skin_seg = (source_Seg[:,SEG.FACE,...] | source_Seg[:,SEG.ARMS,...]| source_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        target_skin_mask = (target_Seg[:,SEG.FACE,...] | target_Seg[:,SEG.ARMS,...]| target_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        
        self.soft_mask_list=[]

        bg_texture, bg_mask = self.segment_encoder(source_Img, bg_Seg, flow)
        face_texture, face_mask = self.segment_encoder(source_Img, face_seg, flow)
        skin_texture, skin_mask = self.segment_encoder(source_Img, skin_seg, flow)
        self.soft_mask_difference(face_mask,target_bg_mask)
        self.soft_mask_difference(face_mask,target_face_mask)
        self.soft_mask_difference(face_mask,target_skin_mask)
        

        # self.soft_mask_list = [bg_mask, face_mask, skin_mask]
        
        batch,channels,h,w = skin_texture.shape
        # skin_avg = skin_texture * skin_mask / (torch.sum(skin_mask) + 0.1)
        # skin_avg = torch.nn.functional.adaptive_avg_pool2d(skin_texture * skin_mask,(1,1))
        # TODO mean skin avg vector over skin mask
        # skin_avg = torch.mean((skin_texture * skin_mask).view(
        #     batch, channels, -1), dim=2).unsqueeze(-1).unsqueeze(-1)
        skin_avg = ((torch.sum((skin_texture * skin_mask).view(batch, channels, -1), dim=2) / 
            (torch.sum((skin_mask).view(batch, channels, -1), dim=2)+0.0001))).unsqueeze(-1).unsqueeze(-1)
        # broadcast skin_avg over body mask
        body_texture = (1 - bg_mask) * skin_avg +  bg_mask * bg_texture + face_mask *  face_texture

        z_pose = self.pose_encoder(target_Pose)
        z_style = self.style_gen(z_pose, body_texture)
        #TODO ad FACE texture

        body_img = self.decoder(z_style.detach()).detach()
        # imsave import cv2;cv2.imwrite('obody.png',127 + 128 * body_img[0].detach().permute(1,2,0).cpu().numpy())
        
        self.masks_validation = {}
        for seg_id in [SEG.HAIR, SEG.PANTS, SEG.SHOES, SEG.UPPER, SEG.DRESS, SEG.HAT]:
            z_style = self.recur_wear(source_Img, source_Seg, target_Seg, seg_id, flow, z_style)
        
        final_img = self.decoder(z_style)
        # imsave import cv2;cv2.imwrite('ofinal.png',127 + 128 * final_img[0].detach().permute(1,2,0).cpu().numpy())
        return flow_fields, body_img, final_img, self.soft_mask_list

####################################################################################################################

class Dior(BaseModel):
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
        self.loss_names = ['correctness_gen', 'regularization_flow', 'l1', 'content_gen', 'style_gen', 'mask',
                            'g_pb', 'g_pp', 'g_ps',
                            'd_pb', 'd_pp', 'd_ps']
        self.loss_names = ['correctness_gen', 'regularization_flow', 'l1', 'content_gen', 'style_gen', 'mask']

        self.visual_names = ['input_P1','input_P2','input_M1','input_BP1', 'input_BP2','input_SP1','input_SP2',
            'img_gen','img_body_gen', 'flow_fields','soft_mask_list'
            ]
        self.model_names = ['G','D_PB','D_PP','D_PS']
        self.model_names = ['G']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        


        # define the generator
        # self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
        #                               layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
        #                               norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        self.net_G = DiorGenerator(opt)
    
        if self.isTrain:
            # pose conditioned
            if opt.with_D_PB:
                self.net_D_PB=init_network(network.discriminator.PatchDiscriminator(
                    input_nc=3+18, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False, use_attn=False
                ), opt)
            # segment conditioned
            if opt.with_D_PS:
                self.net_D_PS=init_network(network.discriminator.PatchDiscriminator(
                    input_nc=3+len(SEG.labels), ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False, use_attn=False
                ), opt)
            # photo conditioned
            if opt.with_D_PP:
                self.net_D_PP=init_network(network.discriminator.PatchDiscriminator(
                    input_nc=3+3, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=opt.use_spect_d,
                    use_coord=False, use_attn=False
                ), opt)

        # # define the discriminator 
        # if self.opt.dataset_mode == 'fashion':
        #     self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        # elif self.opt.dataset_mode== 'market':
        #     self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=3, use_spect=opt.use_spect_d)
        self.flow2color = util.flow2color()

        if self.isTrain:
            # define the loss functions
            # geo loss
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)

            # content loss:l1 + content + style
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # gan loss
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            
            # segmentation loss
            self.SegSoftmaskloss = torch.nn.BCELoss()       
           

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
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
                self.optimizers.append(self.optimizer_DPS)
            
        # load the pre-trained model and schedulers
        self.setup(opt)
        print('---------- Networks initialized -------------')
        print_network(self.net_G)
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


    # def test(self):
    #     """Forward function used in test time"""
    #     img_gen, flow_fields, masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
    #     self.save_results(img_gen, data_name='vis')
    #     result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
    #     self.save_results(result, data_name='all')
    #     if self.opt.save_input or self.opt.phase == 'val':
    #         self.save_results(self.input_P1, data_name='ref')
    #         self.save_results(self.input_P2, data_name='gt')
    #         result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
    #         self.save_results(result, data_name='all')
                       
                

    def forward(self):
        """Run forward processing to get the inputs"""
        self.flow_fields, self.img_body_gen, self.img_gen,self.soft_mask_list = self.net_G(self.input_P1, self.input_BP1, self.input_SP1, self.input_M1, self.input_BP2, self.input_SP2)



    def backward_D_basic(self,  optimizer, netD, real, fake):
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

    # def backward_D(self):
    #     """Calculate the GAN loss for the discriminators"""
    #     base_function._unfreeze(self.net_D)
    #     self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)

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
            D_fake = self.net_D_PB(torch.cat((self.img_gen, self.input_BP2),1))
            self.loss_g_pb = self.GANloss(D_fake, True, False) * self.opt.lambda_gb
        if self.opt.with_D_PS:
            base_function._freeze(self.net_D_PS)
            D_fake = self.net_D_PS(torch.cat((self.img_gen, self.input_SP1),1))
            self.loss_g_ps = self.GANloss(D_fake, True, False) * self.opt.lambda_gs
        if self.opt.with_D_PP:
            base_function._freeze(self.net_D_PP)
            D_fake = self.net_D_PP(torch.cat((self.img_gen, self.input_P1),1))
            self.loss_g_pp = self.GANloss(D_fake, True, False) * self.opt.lambda_gi

        ########## Segmentation mask loss #################
        loss_mask = 0
        for soft_mask_target_mask in self.net_G.soft_mask_list:
            soft_mask, target_mask = soft_mask_target_mask
            loss_mask+=self.SegSoftmaskloss(soft_mask,target_mask)
        # loss_mask = loss_mask / len(self.net_G.soft_mask_list)
        self.loss_mask = loss_mask * self.opt.lambda_seg

        
        
        # total weighted loss
        total_gen_loss = 0
        for name in self.loss_names:
            if not name.startswith('d_') :# and hasattr(self, "loss_" + name)
                total_gen_loss += getattr(self, "loss_" + name)
        total_gen_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

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
                real=torch.cat((self.input_P2, self.input_SP1),1),
                fake=torch.cat((self.img_gen, self.input_SP1),1)
            )

        # if opt.with_D_PB:
        #     self.optimizer_DPP.zero_grad()
        #     self.backward_D()
        #     self.optimizer_DPP.step()
        
        # if opt.with_D_PB:
        #     self.optimizer_DPP.zero_grad()
        #     self.backward_D(self.optimizer_DPP,self.optimizer_DPP)
        #     self.optimizer_DPP.step()  

        


def init_network(net, opt, init_type=None):
    if init_type is None: init_type = opt.init_type
    if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.cuda()
    net.init_weights(init_type)
    return net


def print_network(network):
    if isinstance(network, list):
        network = network[0]
    num_params = 0
    for param in network.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.4f million. '
        'To see the architecture, do print(network).'
        % (type(network).__name__, num_params / 1000000))
    # print(self)
