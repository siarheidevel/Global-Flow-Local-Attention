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


######################################################################################################################################
#Multi encoders full model
class MultiPoseEncoder2(BaseNetwork):
    def __init__(self, input_dim, dim =64, max_dim = 256, n_res = 2, norm='instance', 
                 activation='LeakyReLU', pad_type = 'reflect'):
        super(MultiPoseEncoder2, self).__init__()
        norm_layer = base_function.get_norm_layer(norm)
        nonlinearity = base_function.get_nonlinearity_layer(activation_type=activation)
        self.enc0 = base_function.EncoderBlock(input_dim, dim, norm_layer=norm_layer, nonlinearity=nonlinearity) # 64 h/2
        next_dim = min( max_dim, dim * 2)
        self.enc1 = base_function.EncoderBlock(dim, next_dim, norm_layer=norm_layer, nonlinearity=nonlinearity) # 128 h/4
        self.conv_half1 = nn.Conv2d(next_dim, dim, 1)
        dim = next_dim
        next_dim = min( max_dim, dim * 2)
        self.enc2 = base_function.EncoderBlock(dim, next_dim, norm_layer=norm_layer, nonlinearity=nonlinearity) # 256 h/8
        self.conv_half2 = nn.Conv2d(next_dim, dim, 1)
        dim = next_dim

        self.n_res = n_res
        if self.n_res>0:
            self.res = base_function.ResBlocks(num_blocks=self.n_res, input_nc = next_dim,
                norm_layer = norm_layer)
        self.up = nn.Upsample(scale_factor=2)

        

    def forward(self, x):
        # x  18 * h
        x = self.enc0(x) # 64 * h/2
        x = self.enc1(x) # 128 h/4
        x = self.enc2(x) # 256 h/8
        if self.n_res>0:
            x2 = self.res(x) #256 h/8
        x1 = self.up(x2) #256 h/4
        x1 = self.conv_half2(x1) # 128 h/4
        x0 = self.up(x1) # 128 h/2
        x0 = self.conv_half1(x0) # 64 h/2        
        return x0, x1, x2


class SoftShapeMask(BaseNetwork):
    def __init__(self, dim):
        super(SoftShapeMask, self).__init__()
        self.conv1 = nn.Sequential(nn.InstanceNorm2d(dim),nn.LeakyReLU(),nn.Conv2d(dim,dim,3,1,1,padding_mode='reflect'))
        self.conv2 = nn.Sequential(nn.InstanceNorm2d(dim),nn.LeakyReLU(),nn.Conv2d(dim,1,3,1,1,padding_mode='reflect'))
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.activation(x)
        # return torch.clamp(x,0,1)
        return self.activation(x)


class MultiEncoder2(BaseNetwork):
    def __init__(self, input_img_dim=3,dim=64, style_dim = 256, norm='none', activation='LeakyReLU', pad_type ='reflect'):
        super(MultiEncoder2, self).__init__()
        self.vgg = external_function.VGG19()
        self.conv0 = Conv2dBlock(input_img_dim, dim, 7, 1, 3, norm=norm,activation=activation, pad_type=pad_type)# 3->64,concat
        dim = dim * 2
        # self.enc0 = Conv2dBlock(dim, dim//2, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type) # 128 -> 64
        # self.mask0 = nn.Sequential(*[Conv2dBlock(dim//2,dim//2,3,1,1,norm='instance'),Conv2dBlock(dim//2,1,3,1,1,norm='instance'),nn.Sigmoid()])

        self.conv1 = Conv2dBlock(dim, dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 128->128,concat
        dim = dim * 2
        self.enc1 = Conv2dBlock(dim, dim//4, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type) # 256 -> 64
        self.mask1 = SoftShapeMask(dim//4)

        self.conv2 = Conv2dBlock(dim, dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 256->256,concat
        dim = dim * 2
        self.enc2 = Conv2dBlock(dim, dim//4, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type) # 512 -> 128
        self.mask2 = SoftShapeMask(dim//4)

        self.conv3 = Conv2dBlock(dim, dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 512->512,concat
        dim = dim * 2
        self.enc3 = Conv2dBlock(dim, dim//4, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type) # 1024 -> 256
        self.mask3 = SoftShapeMask(dim//4)

    def flow2img(self, tx, flow):
        if tx.shape[2]!=flow.shape[2] or tx.shape[3]!=flow.shape[3]:
            return F.interpolate(flow, tx.shape[2:], mode='bilinear')
        else:
            return flow

    def forward(self, seg_img, seg_mask, flow):
        xi = seg_img * seg_mask.repeat(1, seg_img.size(1), 1, 1)
        _,_, f_height,f_width = flow.shape
        sty_fea = self.vgg(xi)
        # import cv2;cv2.imwrite('oo2.png',127 + 127 * torch.cat((xi,util.bilinear_warp(xi, F.interpolate(flow,(xi.shape[2],xi.shape[3])))),-1)[0].permute(1,2,0).cpu().detach().numpy())
        x = self.conv0(xi) # 64 * h
        x = torch.cat([x, sty_fea['relu1_1'].detach()], dim=1) # 128 * h
        # tx0 = self.enc0(x)
        # tx0 = util.bilinear_warp(tx0, self.flow2img(tx0,flow))
        # mx0 = self.mask0(tx0)

        x = self.conv1(x)
        x = torch.cat([x, sty_fea['relu2_1'].detach()], dim=1) # 256 * h/2
        tx1 = self.enc1(x)
        tx1 = util.bilinear_warp(tx1, self.flow2img(tx1,flow))
        mx1 = self.mask1(tx1)

        x = self.conv2(x)
        x = torch.cat([x, sty_fea['relu3_1'].detach()], dim=1) # 512 * h/4
        tx2 = self.enc2(x)
        tx2 = util.bilinear_warp(tx2, self.flow2img(tx2,flow))
        mx2 = self.mask2(tx2)

        x = self.conv3(x)
        x = torch.cat([x, sty_fea['relu4_1'].detach()], dim=1) # 1024 * h/8
        tx3 = self.enc3(x)
        tx3 = util.bilinear_warp(tx3, self.flow2img(tx3,flow))
        mx3 = self.mask3(tx3)

        # import cv2;cv2.imwrite('oo1.png',254 * sem_mask.type(torch.float)[0][0].cpu().numpy())
        # import cv2;cv2.imwrite('oo2s.png',254 * torch.cat((util.bilinear_warp(sem_mask.type(torch.float), flow),F.interpolate(mx,sem_mask.shape[2:])),-1)[0][0].detach().cpu().numpy())
        return (tx1, tx2, tx3), (mx1, mx2, mx3)

class MultiDecoder2(BaseNetwork):
    def __init__(self, input_dim, output_dim, norm = 'layer'):
        super(MultiDecoder2, self).__init__()
        self.model = []
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = Conv2dBlock(input_dim, input_dim//2, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')
        self.conv1 = Conv2dBlock(input_dim, input_dim//4, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')
        
        self.final_conv = Conv2dBlock(input_dim//2, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')
    
    def forward(self, x:list , detach_grad:bool = False):
        x0, x1, x2 = x[0],x[1],x[2]
        if detach_grad:
            x0 = x0.detach()
            x1 = x1.detach()
            x2 = x2.detach()
        # x2 - L * h/8
        # x1 - L/2 * h/4
        # x0 - L/4 * h/2
        x = self.up(x2) # L * h/4
        x = self.conv2(x) # L/2 * h/4   
        
        x = torch.cat([x, x1], dim=1) # L * h/4
        x = self.up(x) # L * h/2
        x = self.conv1(x) # L/4 * h/2

        x = torch.cat([x, x0], dim=1) # L/2 * h/2
        x = self.up(x) # L * h/2
        out = self.final_conv(x) # L * 3
        if detach_grad:
            out = out.detach()
        return out

class MultiStyle2(BaseNetwork):
    def __init__(self, style_dim=256, spade_config_str ='spadeinstance3x3',use_spectral= True, learned_shortcut =True):
        super(MultiStyle2, self).__init__()
        self.up = nn.Upsample(scale_factor=2)         
        self.sp0 = base_function.SPADEResnetBlock(input_nc = style_dim//4, output_nc = style_dim//4,
                                                          hidden_nc = style_dim//4, label_nc =style_dim//4,
                                                          spade_config_str=spade_config_str,use_spect=use_spectral,
                                                          learned_shortcut=learned_shortcut)
        self.sp1 = base_function.SPADEResnetBlock(input_nc = style_dim//2, output_nc = style_dim//2,
                                                          hidden_nc = style_dim//2, label_nc =style_dim//2,
                                                          spade_config_str=spade_config_str,use_spect=use_spectral,
                                                          learned_shortcut=learned_shortcut)
        self.sp2 = base_function.SPADEResnetBlock(input_nc = style_dim, output_nc = style_dim,
                                                          hidden_nc = style_dim, label_nc =style_dim,
                                                          spade_config_str=spade_config_str,use_spect=use_spectral,
                                                          learned_shortcut=learned_shortcut)
        

    def forward(self, prev_states, textures, masks):
        state0 = prev_states[0] # L/4 * H/2
        state1 = prev_states[1] # L/2 * H/4
        state2 = prev_states[2] # L * H/8

        tx0 = textures[0] # L/4 * H/2
        tx1 = textures[1] # L/2 * H/4
        tx2 = textures[2] # L * H/8

        m0 = masks[0] # 1 * H/2
        m1 = masks[1] # 1 * H/4
        m2 = masks[2] # 1 * H/8
        
        style0 = state0*(1 -m0) + self.sp0(state0, tx0) * m0
        style1 = state1*(1 -m1) + self.sp1(state1, tx1) * m1
        style2 = state2*(1 -m2) + self.sp2(state2, tx2) * m2
        
        # TODO cross layer links

        return [style0,style1, style2]

class MultiDiorGenerator2(BaseNetwork):
    def __init__(self,image_nc=3, structure_nc=18, norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False):
        super(MultiDiorGenerator2, self).__init__()
        self.pose_encoder = MultiPoseEncoder2(input_dim=18,dim=64, max_dim=256, n_res=5,norm='instance')
        self.style_gen = MultiStyle2(style_dim = 256, spade_config_str='spadeinstance3x3', use_spectral=use_spect, learned_shortcut=True)    
    
    
    @staticmethod
    def soft_mask_difference(soft_mask_list, mask, seg):
        for i in range(len(mask)):
            soft_mask_list.append([mask[i],F.interpolate(seg.float(),mask[i].shape[2:])])
        
    
    def create_body_texture(self,bg_textures, bg_masks, face_textures, face_masks, skin_textures, skin_masks):
        body_textures=[]
        body_masks =[]
        for i in range(len(bg_textures)):
            batch,channels,h,w = skin_textures[i].shape
            # avg vector over mask region
            skin_avg = ((torch.sum((skin_textures[i] * skin_masks[i]).view(batch, channels, -1), dim=2) / 
                (torch.sum((skin_masks[i]).view(batch, 1, -1), dim=2)+0.00001))).unsqueeze(-1).unsqueeze(-1)
            # skin_avg = torch.nn.AdaptiveAvgPool2d(1)(skin_texture * skin_mask)
            # broadcast skin_avg over body mask
            body_texture = (1 - bg_masks[i]) * skin_avg +  bg_masks[i] * bg_textures[i] + face_masks[i] *  face_textures[i]
            body_textures.append(body_texture)
            body_masks.append(torch.ones_like(body_texture))
        return  body_textures, body_masks


    def forward(self, segment_encoder: MultiEncoder2,
                image_decoder: MultiDecoder2,
                flow_generator,
                source_Img, source_Pose, source_Seg, target_Pose, target_Seg):
        from data.dior2_dataset import SEG


        flow_fields = flow_generator(source_Img, source_Pose, target_Pose)[0]
        # take h/4, w/4 flow                
        flow = flow_fields[-1]
        # # masking for inpainting        
        # source_Img = source_Img * source_Mask.unsqueeze(1)
        # source_Seg = source_Seg * source_Mask.unsqueeze(1)

        bg_Seg = source_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        target_bg_mask = target_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        face_seg = source_Seg[:,SEG.FACE,...].unsqueeze(1)
        target_face_mask = target_Seg[:,SEG.FACE,...].unsqueeze(1)
        skin_seg = (source_Seg[:,SEG.FACE,...] | source_Seg[:,SEG.ARMS,...]| source_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        target_skin_mask = (target_Seg[:,SEG.FACE,...] | target_Seg[:,SEG.ARMS,...]| target_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        
        soft_mask_list=[]
        intermediate_images = []
        bg_textures, bg_masks = segment_encoder(source_Img, bg_Seg, flow)
        face_textures, face_masks = segment_encoder(source_Img, face_seg, flow)
        skin_textures, skin_masks = segment_encoder(source_Img, skin_seg, flow)
        MultiDiorGenerator2.soft_mask_difference(soft_mask_list, bg_masks, target_bg_mask)
        MultiDiorGenerator2.soft_mask_difference(soft_mask_list, face_masks, target_face_mask)
        MultiDiorGenerator2.soft_mask_difference(soft_mask_list, skin_masks, target_skin_mask)
        
        
        body_textures, body_masks=self.create_body_texture(bg_textures,bg_masks, face_textures,face_masks, skin_textures,skin_masks)

        z_pose = self.pose_encoder(target_Pose)
        z_styles = self.style_gen(z_pose, body_textures, body_masks)
        #TODO ad FACE texture

        body_img = image_decoder(z_styles, detach_grad=True)
        intermediate_images.append(body_img.cpu())
        # imsave import cv2;cv2.imwrite('obody.png',127 + 128 * body_img[0].detach().permute(1,2,0).cpu().numpy())
        
        for seg_id in [SEG.HAIR, SEG.SHOES, SEG.PANTS, SEG.UPPER, SEG.HAT]:
            seg = source_Seg[:,seg_id,...].unsqueeze(1)
            t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
            garment_textures, garment_masks = segment_encoder(source_Img, seg, flow)
            MultiDiorGenerator2.soft_mask_difference(soft_mask_list, garment_masks, t_seg)
            z_styles = self.style_gen(z_styles, garment_textures, garment_masks)
            # for i in range(len(z_style_next)):
            #         z_style[i] = z_style_next[i] * garment_mask[i] + z_style[i] * (1 - garment_mask[i])
            garment_img = image_decoder(z_styles, detach_grad=True)
            intermediate_images.append(garment_img.cpu())
        
        final_img = image_decoder(z_styles)
        # imsave import cv2;cv2.imwrite('ofinal.png',127 + 128 * final_img[0].detach().permute(1,2,0).cpu().numpy())
        return flow_fields, final_img, soft_mask_list, intermediate_images



