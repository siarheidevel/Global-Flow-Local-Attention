import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function, BaseNetwork
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
    def __init__(self, input_dim, style_dim, inner_dim =64, norm='none', activation='LeakyReLU', pad_type ='reflect'):
        super(SegmentEncoder, self).__init__()
        self.add_module('vgg', external_function.VGG19())
        self.conv1 = Conv2dBlock(input_dim, inner_dim, 7, 1, 3, norm=norm,activation=activation, pad_type=pad_type)# 3->64,concat
        inner_dim = inner_dim * 2
        self.conv2 = Conv2dBlock(inner_dim, inner_dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 128->128,concat
        inner_dim = inner_dim * 2
        self.conv3 = Conv2dBlock(inner_dim, inner_dim, 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 256->256,concat
        inner_dim = inner_dim * 2
        # self.conv4 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 512->512,concat
        # dim = dim * 2
        self.enc = nn.Sequential(*[nn.Conv2d(inner_dim, style_dim, 1, 1, 0)])
        # self.bilinear_warp = base_function.BilinearSamplingBlock()
        # self.soft_mask = SoftShapeMask(style_dim,norm,num_blocks=2)
        self.soft_mask = nn.Sequential(*[base_function.ResBlock(style_dim,1,norm_layer=None),nn.Sigmoid()])


    def forward(self, seg_img, seg_mask, flow):
        xi = seg_img * seg_mask.repeat(1, seg_img.size(1), 1, 1)
        # import cv2;cv2.imwrite('oo2.png',127 + 127 * torch.cat((xi,util.bilinear_warp(xi, F.interpolate(flow,(xi.shape[2],xi.shape[3])))),-1)[0].permute(1,2,0).cpu().detach().numpy())
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
                 activation='LeakyReLU', pad_type = 'reflect'):
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


class StyleEmbedder(BaseNetwork):
    def __init__(self,n_blocks = 6,style_dim=128, spade_config_str ='spadeinstance3x3'):
        super(StyleEmbedder, self).__init__()
        # input_nc, output_nc, hidden_nc, label_nc, spade_config_str='spadeinstance3x3', nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False, learned_shortcut=False
        self.blocks = []
        for _ in range(n_blocks):
            self.blocks += [base_function.SPADEResnetBlock(input_nc = style_dim, output_nc = style_dim,
                                                          hidden_nc = style_dim, label_nc =style_dim,
                                                          spade_config_str=spade_config_str)]
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, prev_state, texture):
        state = prev_state
        for _, block in enumerate(self.blocks):
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
            # self.model += [base_function.ResBlock(input_dim, input_dim//2,
            #     learnable_shortcut=True, norm_layer=base_function.get_norm_layer(norm))]
            self.model += [Conv2dBlock(input_dim, input_dim//2, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')]
            input_dim = input_dim // 2
        self.model += [Conv2dBlock(input_dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)


class DiorGenerator(BaseNetwork):
    def __init__(self,image_nc=3, structure_nc=18, style_dim= 128, style_blocks=8,norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False):
        super(DiorGenerator, self).__init__()
        # flow field generator
        # self.flow_net =network.generator.PoseFlowNet(image_nc=3, structure_nc=18, ngf=32, img_f=256,
        #                 encoder_layer=5, attn_layer=[1,2],
        #                 norm='instance', activation='LeakyReLU',
        #                 use_spect=opt.use_spect_g, use_coord=False)
        self.pose_encoder= PoseEncoder(n_downsample=2,input_dim=18,dim=64, max_dim=style_dim, n_res=8,norm='instance')        
        # self.style_embedder = StyleEmbedder(n_blocks=style_blocks, style_dim=style_dim, spade_config_str='spadeinstance3x3')
        self.body_gen = base_function.SPADEResnetBlock(input_nc = style_dim, output_nc = style_dim,
                                                          hidden_nc = style_dim, label_nc =style_dim,
                                                          spade_config_str='spadeinstance3x3')
        self.garment_gen = base_function.SPADEResnetBlock(input_nc = style_dim, output_nc = style_dim,
                                                          hidden_nc = style_dim, label_nc =style_dim,
                                                          spade_config_str='spadeinstance3x3')
    
    def config_net(self, segment_encoder: SegmentEncoder, image_decoder: Decoder, flow_generator):
        self.segment_encoder = segment_encoder
        self.image_decoder = image_decoder
        self.flow_generator = flow_generator
    

    # def recur_wear(self, img, source_Seg, target_Seg, seg_id, flow, prev_style):
    #     seg = source_Seg[:,seg_id,...].unsqueeze(1)
    #     t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
    #     _texture, _mask = self.segment_encoder(img, seg, flow)
    #     self.soft_mask_difference(_mask,t_seg)
    #     # warped_seg = util.bilinear_warp(seg.type(torch.float),flow)
    #     # seg_small = F.interpolate(warped_seg,_mask.shape[-2:]).detach()
    #     # warp seg and interpolate import cv2;cv2.imwrite('oo_s.png',254 * torch.cat((seg_small,_mask),-1)[0][0].cpu().detach().numpy())
    #     # self.masks_validation[seg_id]=(_mask,seg_small)
    #     z_style_next = self.style_embedder(prev_style, _texture)
    #     z_style_next = z_style_next * _mask + prev_style * (1 - _mask)
    #     # self.soft_mask_list.append(torch.cat((_mask,seg_small),-1))
        
    #     return z_style_next


    # def flow_estimate(self,source_Img, source_Pose, target_Pose):
    #     flow_fields = self.flow_net(source_Img, source_Pose, target_Pose)
    #     return flow_fields[0][0]
    
    @staticmethod
    def soft_mask_difference( soft_mask_list, mask, seg):
        soft_mask_list.append([mask,F.interpolate(seg.float(),mask.shape[2:])])


    def forward(self, source_Img, source_Pose, source_Seg, source_Mask, target_Pose, target_Seg):
        flow_fields = self.flow_generator(source_Img, source_Pose, target_Pose)[0]
        # take h/4, w/4 flow        
        flow = flow_fields[-1]

        # masking for inpainting        
        source_Img = source_Img * source_Mask.unsqueeze(1)
        source_Seg = source_Seg * source_Mask.unsqueeze(1)

        bg_Seg = source_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        target_bg_mask = target_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        face_seg = source_Seg[:,SEG.FACE,...].unsqueeze(1)
        target_face_mask = target_Seg[:,SEG.FACE,...].unsqueeze(1)
        skin_seg = (source_Seg[:,SEG.FACE,...] | source_Seg[:,SEG.ARMS,...]| source_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        target_skin_mask = (target_Seg[:,SEG.FACE,...] | target_Seg[:,SEG.ARMS,...]| target_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        
        soft_mask_list=[]

        bg_texture, bg_mask = self.segment_encoder(source_Img, bg_Seg, flow)
        face_texture, face_mask = self.segment_encoder(source_Img, face_seg, flow)
        skin_texture, skin_mask = self.segment_encoder(source_Img, skin_seg, flow)
        DiorGenerator.soft_mask_difference(soft_mask_list, bg_mask, target_bg_mask)
        DiorGenerator.soft_mask_difference(soft_mask_list, face_mask, target_face_mask)
        DiorGenerator.soft_mask_difference(soft_mask_list, skin_mask, target_skin_mask)
        

        # self.soft_mask_list = [bg_mask, face_mask, skin_mask]
        
        batch,channels,h,w = skin_texture.shape
        # avg vector over mask region
        skin_avg = ((torch.sum((skin_texture * skin_mask).view(batch, channels, -1), dim=2) / 
            (torch.sum((skin_mask).view(batch, 1, -1), dim=2)+0.00001))).unsqueeze(-1).unsqueeze(-1)
        # skin_avg = torch.nn.AdaptiveAvgPool2d(1)(skin_texture * skin_mask)
        # broadcast skin_avg over body mask
        body_texture = (1 - bg_mask) * skin_avg +  bg_mask * bg_texture + face_mask *  face_texture

        z_pose = self.pose_encoder(target_Pose)
        z_style = self.body_gen(z_pose, body_texture)
        #TODO ad FACE texture

        body_img = self.image_decoder(z_style.detach()).detach()
        # imsave import cv2;cv2.imwrite('obody.png',127 + 128 * body_img[0].detach().permute(1,2,0).cpu().numpy())
        
        for seg_id in [SEG.HAIR, SEG.SHOES, SEG.PANTS, SEG.UPPER, SEG.DRESS, SEG.HAT]:
            seg = source_Seg[:,seg_id,...].unsqueeze(1)
            t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
            garment_texture, garment_mask = self.segment_encoder(source_Img, seg, flow)
            DiorGenerator.soft_mask_difference(soft_mask_list, garment_mask, t_seg)
            z_style_next = self.garment_gen(z_style, garment_texture) * garment_mask + z_style * (1 - garment_mask)
            z_style = z_style_next
        
        final_img = self.image_decoder(z_style)
        # imsave import cv2;cv2.imwrite('ofinal.png',127 + 128 * final_img[0].detach().permute(1,2,0).cpu().numpy())
        return flow_fields, body_img, final_img, soft_mask_list


#################################################################################################################
class MultiSegmentEncoder(BaseNetwork):
    def __init__(self, input_dim, style_dim, inner_dim =64, norm='none', activation='LeakyReLU', pad_type ='reflect'):
        super(MultiSegmentEncoder, self).__init__()
        
        half_style_dim = style_dim // 2

        self.add_module('vgg', external_function.VGG19())
        self.conv1 = Conv2dBlock(input_dim, inner_dim, 7, 1, 3, norm=norm,activation=activation, pad_type=pad_type)# 3->64,concat
        inner_dim = inner_dim * 2
        self.conv2 = Conv2dBlock(inner_dim, inner_dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 128->128,concat
        inner_dim = inner_dim * 2

        self.enc1 = nn.Sequential(*[nn.Conv2d(inner_dim, half_style_dim, 1, 1, 0)])
        self.soft_mask1 = nn.Sequential(*[base_function.ResBlock(half_style_dim,1,norm_layer=None),nn.Sigmoid()])

        self.conv3 = Conv2dBlock(inner_dim, inner_dim, 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 256->256,concat
        inner_dim = inner_dim * 2
        # self.conv4 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 512->512,concat
        # dim = dim * 2
        self.enc2 = nn.Sequential(*[nn.Conv2d(inner_dim, style_dim, 1, 1, 0)])
        self.soft_mask2 = nn.Sequential(*[base_function.ResBlock(style_dim,1,norm_layer=None),nn.Sigmoid()])


    def forward(self, seg_img, seg_mask, flows):
        flow1 = flows[1] #h/2
        flow2 = flows[0] #h/4
        xi = seg_img * seg_mask.repeat(1, seg_img.size(1), 1, 1)
        # import cv2;cv2.imwrite('oo2.png',127 + 127 * torch.cat((xi,util.bilinear_warp(xi, F.interpolate(flow,(xi.shape[2],xi.shape[3])))),-1)[0].permute(1,2,0).cpu().detach().numpy())
        sty_fea = self.vgg(xi)

        x = self.conv1(xi) # 64 * h
        x = torch.cat([x, sty_fea['relu1_1']], dim=1) # 128 * h
        x = self.conv2(x) # 128 * h/2
        x = torch.cat([x, sty_fea['relu2_1']], dim=1) # 256 * h/2

        tx1  = self.enc1(x)
        tx1 = util.bilinear_warp(tx1, flow1)
        mx1 = self.soft_mask1(tx1)        

        x = self.conv3(x) # 256 * h/4
        x = torch.cat([x, sty_fea['relu3_1']], dim=1) # 512 * h/4
        # x = self.conv4(x)
        # x = torch.cat([x, sty_fea['relu4_1']], dim=1)
        tx2 = self.enc2(x)
        # warp by flow field using bilinear interpolation
        tx2 = util.bilinear_warp(tx2, flow2)
        # soft mask torch.nn.BCELoss()(mx[0][0],torch.nn.functional.interpolate(sem_mask.type(torch.float),size=mx.shape[-2:])[0][0])
        # nn.Sequential(*[nn.Conv2d(128,64,3,1,1), nn.LeakyReLU(),nn.Conv2d(64,1,3,1,1),nn.Sigmoid()])
        # import cv2;cv2.imwrite('oo1.png',254 * sem_mask.type(torch.float)[0][0].cpu().numpy())
        # import cv2;cv2.imwrite('oo2s.png',254 * torch.cat((util.bilinear_warp(sem_mask.type(torch.float), flow),F.interpolate(mx,sem_mask.shape[2:])),-1)[0][0].detach().cpu().numpy())
        mx2 = self.soft_mask2(tx2)

        # segment loss and view
        # import cv2;cv2.imwrite('oo_s.png',254 * torch.cat((seg_small,_mask),-1)[0][0].cpu().detach().numpy())
        # warped_seg = util.bilinear_warp(sem_mask.type(torch.float),flow)
        # seg_small = F.interpolate(warped_seg,mx.shape[-2:]).detach()
        # soft_mask_list.append(torch.cat((mx,seg_small),-1))
        return (tx1, tx2), (mx1, mx2)


class MultiDecoder(BaseNetwork):
    def __init__(self, input_dim, output_dim, norm = 'layer'):
        super(MultiDecoder, self).__init__()
        self.model = []
        self.up = nn.Upsample(scale_factor=2)

        self.conv1 = Conv2dBlock(input_dim, input_dim//2, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')
        self.conv2 = Conv2dBlock(input_dim, input_dim//2, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')
        self.final_conv = Conv2dBlock(input_dim//2, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')
    
    def forward(self, x1, x2, detach_grad:bool = False):
        if detach_grad:
            x1 = x1.detach()
            x2 = x2.detach()
        # x2 - L * h/4
        # x1 - L/2 * h/2
        x = self.up(x2) # L * h/2
        x = self.conv1(x) # L/2 * h/2   
        
        x = torch.cat([x, x1], dim=1) # L * h/2
        x = self.up(x) # L * h

        x = self.conv2(x) # L/2 *h
        out = self.final_conv(x)
        if detach_grad:
            out = out.detach()
        return out

class MultiPoseEncoder(BaseNetwork):
    def __init__(self, input_dim, dim =64, max_dim = 256, n_res = 2, norm='instance', 
                 activation='LeakyReLU', pad_type = 'reflect'):
        super(MultiPoseEncoder, self).__init__()
        self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)
        next_dim = min( max_dim, dim * 2)
        self.conv2 = Conv2dBlock(dim, next_dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type) # L/2 h/2
        dim = min( max_dim, dim * 2)
        next_dim = min( max_dim, dim * 2)
        self.conv3 = Conv2dBlock(dim, next_dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type) # L/2 h/2
        dim = min( max_dim, dim * 2)
        next_dim = min( max_dim, dim * 2)
        self.n_res = n_res
        if self.n_res>0:
            self.res1 = base_function.ResBlocks(num_blocks=self.n_res, input_nc = dim,
                norm_layer = base_function.get_norm_layer(norm))
        self.up = nn.Upsample(scale_factor=2)
        self.conv_half = nn.Conv2d(dim,dim//2, 1) # L h/4        
        

    def forward(self, x):
        # x  18 * h
        x = self.conv1(x) # 64 * h
        x = self.conv2(x) # 128 * h/2
        x = self.conv3(x) # 128 * h/4
        if self.n_res>0:
            x2 = self.res1(x)
        x1 = self.conv_half(self.up(x2)) # 64 * h/2        
        return x1, x2


class MultiStyleEmbedder(BaseNetwork):
    def __init__(self, style_dim=128, spade_config_str ='spadeinstance3x3',use_spectral= True):
        super(MultiStyleEmbedder, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.sp2 = base_function.SPADEResnetBlock(input_nc = style_dim, output_nc = style_dim,
                                                          hidden_nc = style_dim, label_nc =style_dim,
                                                          spade_config_str=spade_config_str,use_spect=use_spectral)
        
        self.sp1 = base_function.SPADEResnetBlock(input_nc = style_dim//2, output_nc = style_dim//2,
                                                          hidden_nc = style_dim//2, label_nc =style_dim//2,
                                                          spade_config_str=spade_config_str,use_spect=use_spectral)

    def forward(self, prev_states, textures):
        state1 = prev_states[0] # L/2 * H/2
        state2 = prev_states[1] # L * H/4

        tx1 = textures[0] # L/2 * H/2
        tx2 = textures[1] # L * H/4
        
        style2 = self.sp2(state2, tx2)
        style1 = self.sp1(state1, tx1)

        return [style1, style2]


class MultiDiorGenerator(BaseNetwork):
    def __init__(self,image_nc=3, structure_nc=18, style_dim= 128, norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False):
        super(MultiDiorGenerator, self).__init__()
        # flow field generator
        # self.flow_net =network.generator.PoseFlowNet(image_nc=3, structure_nc=18, ngf=32, img_f=256,
        #                 encoder_layer=5, attn_layer=[1,2],
        #                 norm='instance', activation='LeakyReLU',
        #                 use_spect=opt.use_spect_g, use_coord=False)
        self.pose_encoder = MultiPoseEncoder(input_dim=18,dim=64, max_dim=style_dim, n_res=4,norm='instance')        
        self.style_gen = MultiStyleEmbedder(style_dim, spade_config_str='spadeinstance3x3', use_spectral=use_spect)
    
    
    
    @staticmethod
    def soft_mask_difference(soft_mask_list, mask, seg):
        for i in range(len(mask)):
            soft_mask_list.append([mask[i],F.interpolate(seg.float(),mask[i].shape[2:])])
        
    
    def create_body_texture(self,bg_textures, bg_masks, face_textures, face_masks, skin_textures, skin_masks):
        body_textures=[]
        for i in range(len(bg_textures)):
            batch,channels,h,w = skin_textures[i].shape
            # avg vector over mask region
            skin_avg = ((torch.sum((skin_textures[i] * skin_masks[i]).view(batch, channels, -1), dim=2) / 
                (torch.sum((skin_masks[i]).view(batch, 1, -1), dim=2)+0.00001))).unsqueeze(-1).unsqueeze(-1)
            # skin_avg = torch.nn.AdaptiveAvgPool2d(1)(skin_texture * skin_mask)
            # broadcast skin_avg over body mask
            body_texture = (1 - bg_masks[i]) * skin_avg +  bg_masks[i] * bg_textures[i] + face_masks[i] *  face_textures[i]
            body_textures.append(body_texture)
        return  body_textures


    def forward(self, segment_encoder: MultiSegmentEncoder, image_decoder: MultiDecoder, flow_generator,
                source_Img, source_Pose, source_Seg, source_Mask, target_Pose, target_Seg):
        from data.dior2_dataset import SEG


        flow_fields = flow_generator(source_Img, source_Pose, target_Pose)[0]
        # take h/4, w/4 flow                
        flows = flow_fields[1:]
        # masking for inpainting        
        source_Img = source_Img * source_Mask.unsqueeze(1)
        source_Seg = source_Seg * source_Mask.unsqueeze(1)

        bg_Seg = source_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        target_bg_mask = target_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        face_seg = source_Seg[:,SEG.FACE,...].unsqueeze(1)
        target_face_mask = target_Seg[:,SEG.FACE,...].unsqueeze(1)
        skin_seg = (source_Seg[:,SEG.FACE,...] | source_Seg[:,SEG.ARMS,...]| source_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        target_skin_mask = (target_Seg[:,SEG.FACE,...] | target_Seg[:,SEG.ARMS,...]| target_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        
        soft_mask_list=[]
        intermediate_images = []
        bg_texture, bg_mask = segment_encoder(source_Img, bg_Seg, flows)
        face_texture, face_mask = segment_encoder(source_Img, face_seg, flows)
        skin_texture, skin_mask = segment_encoder(source_Img, skin_seg, flows)
        MultiDiorGenerator.soft_mask_difference(soft_mask_list, bg_mask, target_bg_mask)
        MultiDiorGenerator.soft_mask_difference(soft_mask_list, face_mask, target_face_mask)
        MultiDiorGenerator.soft_mask_difference(soft_mask_list, skin_mask, target_skin_mask)
        
        
        body_texture=self.create_body_texture(bg_texture,bg_mask, face_texture,face_mask, skin_texture,skin_mask)

        z_pose = self.pose_encoder(target_Pose)
        z_style = self.style_gen(z_pose, body_texture)
        #TODO ad FACE texture

        body_img = image_decoder(*z_style, detach_grad=True)
        intermediate_images.append(body_img.cpu())
        # imsave import cv2;cv2.imwrite('obody.png',127 + 128 * body_img[0].detach().permute(1,2,0).cpu().numpy())
        
        for seg_id in [SEG.HAIR, SEG.SHOES, SEG.PANTS, SEG.UPPER, SEG.HAT]:
            seg = source_Seg[:,seg_id,...].unsqueeze(1)
            t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
            garment_texture, garment_mask = segment_encoder(source_Img, seg, flows)
            MultiDiorGenerator.soft_mask_difference(soft_mask_list, garment_mask, t_seg)
            z_style_next = self.style_gen(z_style, garment_texture)
            for i in range(len(z_style_next)):
                    z_style[i] = z_style_next[i] * garment_mask[i] + z_style[i] * (1 - garment_mask[i])
            garment_img = image_decoder(*z_style, detach_grad=True)
            intermediate_images.append(garment_img.cpu())
        
        final_img = image_decoder(*z_style)
        # imsave import cv2;cv2.imwrite('ofinal.png',127 + 128 * final_img[0].detach().permute(1,2,0).cpu().numpy())
        return flow_fields, final_img, soft_mask_list, intermediate_images

