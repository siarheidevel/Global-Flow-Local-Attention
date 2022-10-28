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
                 padding=0,dilation=1, norm='none', activation='ReLU', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.pad = base_function.get_padding(pad_type, padding)()
        kwargs_conv = {'kernel_size': kernel_size, 'stride': stride, 'dilation':dilation}
        self.conv = base_function.coord_conv(input_dim, output_dim, **kwargs_conv)
        if activation!='none':
            self.activation = base_function.get_nonlinearity_layer(activation)
        if norm != 'none':
            self.norm = base_function.get_norm_layer(norm)(output_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if hasattr(self,'norm'):
            x = self.norm(x)
        if hasattr(self,'activation'):
            x = self.activation(x)
        return x


######################################################################################################################################
#Multi encoders full model
class PoseEncoder2(BaseNetwork):
    def __init__(self, input_dim, dim =64, max_dim = 256, n_res = 1, norm='instance', 
                 activation='LeakyReLU', pad_type = 'reflect'):
        super(PoseEncoder2, self).__init__()
        norm_layer = base_function.get_norm_layer(norm)
        nonlinearity = base_function.get_nonlinearity_layer(activation_type=activation)
        self.conv0 = nn.Conv2d(input_dim, dim, 5, 1, 2, padding_mode='zeros')
        next_dim = min( max_dim, dim * 2)
        self.enc0 = base_function.EncoderBlock(dim, next_dim, norm_layer=None, nonlinearity=nonlinearity) # 128 h/2
        dim = next_dim
        next_dim = min( max_dim, dim * 2)
        self.enc1 = base_function.EncoderBlock(dim, next_dim, norm_layer=norm_layer, nonlinearity=nonlinearity) # 256 h/4       
        self.n_res = n_res
        if self.n_res>0:
            self.res = nn.Sequential(*[base_function.ResBlock(next_dim,next_dim,norm_layer=norm_layer, nonlinearity=nonlinearity)  for i in range(self.n_res)])
            

    def forward(self, x):
        # x  18 * h
        x = self.conv0(x) # 64 * h/2
        x = self.enc0(x) # 128 * h/2
        x = self.enc1(x) # 256 h/4
        if self.n_res>0:
            x = self.res(x) #256 h/4      
        return x


class SoftShapeMask(BaseNetwork):
    def __init__(self, dim):
        super(SoftShapeMask, self).__init__()
        self.conv1 = Conv2dBlock(dim, 128 , 1, 1, 0, norm='none',activation='LeakyReLU', pad_type='zero')
        self.conv1_1 = Conv2dBlock(128, 64 , 3, 1, 1,dilation=1, norm='none',activation='LeakyReLU', pad_type='zero')
        self.conv1_2 = Conv2dBlock(128, 64 , 3, 1, 2,dilation=2, norm='none',activation='LeakyReLU', pad_type='zero')
        self.conv1_3 = Conv2dBlock(128, 64 , 3, 1, 3,dilation=3, norm='none',activation='LeakyReLU', pad_type='zero')
        self.conv2 = Conv2dBlock(192, 1 , 3, 1, 1, norm='none',activation='none', pad_type='zero')
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x_1 = self.conv1_1(x)
        x_2 = self.conv1_2(x)
        x_3 = self.conv1_3(x)
        
        x = self.conv2(torch.cat((x_1,x_2,x_3),1))
        x = self.activation(x)
        # torch.mean(x),torch.std(x),torch.min(x),torch.max(x)
        # return torch.clamp(x,0,1)
        return x


class Fusion(BaseNetwork):
    def __init__(self,input_dim, output_dim, inner_dim=None,layers=1,norm='none', activation='LeakyReLU', pad_type ='reflect'):
        super(Fusion, self).__init__()
        if inner_dim==None:
            inner_dim=output_dim
        self.layers=layers
        self.conv_in = Conv2dBlock(input_dim, inner_dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
        if self.layers>0:
            norm_layer = base_function.get_norm_layer(norm)
            nonlinearity = base_function.get_nonlinearity_layer(activation_type=activation)
            self.inner_conv = nn.Sequential(*[base_function.ResBlock(
                inner_dim,inner_dim,norm_layer=norm_layer, nonlinearity=nonlinearity)  for i in range(self.layers)])
        self.conv_out=Conv2dBlock(inner_dim, output_dim*4, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type) 
        
    def forward(self, x):
        x = self.conv_in(x)
        if self.layers>0:
            x = self.inner_conv(x)
        x = self.conv_out(x)
        return x

class Encoder2(BaseNetwork):
    def __init__(self, input_img_dim=3,dim=64, style_dim = 256, norm='none', activation='LeakyReLU', pad_type ='reflect'):
        super(Encoder2, self).__init__()
        # self.vgg = external_function.VGG19()
        self.conv0 = Conv2dBlock(input_img_dim, dim, 7, 1, 3, norm=norm,activation=activation, pad_type=pad_type)# 3->64,concat
        dim = dim * 2
        self.conv1 = Conv2dBlock(dim, dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 128->128,concat
        dim = dim * 2        
        self.conv2 = Conv2dBlock(dim, dim , 4, 2, 1, norm=norm,activation=activation, pad_type=pad_type)# 256->256,concat
        # dim = dim * 2
        # self.enc2 = Conv2dBlock(dim, style_dim*4, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type) # 512 -> 1024
        # self.enc2= nn.Sequential(
        #     Conv2dBlock(2*dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type) # 512 -> 256
        #     ,Conv2dBlock(dim, style_dim*4, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type) # 512 -> 1024
        # )
        self.enc2 = Fusion(2*dim, style_dim, inner_dim=256,layers=1)
        self.mask2 = SoftShapeMask(style_dim*4)

    def flow2img(self, tx, flow):
        if tx.shape[2]!=flow.shape[2] or tx.shape[3]!=flow.shape[3]:
            return F.interpolate(flow, tx.shape[2:], mode='bilinear')
        else:
            return flow

    def forward(self, vgg, seg_img, seg_mask, flow):
        xi = seg_img * seg_mask.repeat(1, seg_img.size(1), 1, 1)
        # _,_, f_height,f_width = flow.shape
        with torch.no_grad():
            sty_fea = vgg(xi, max_layer='relu3_1')
        # import cv2;cv2.imwrite('oo2.png',127 + 127 * torch.cat((xi,util.bilinear_warp(xi, F.interpolate(flow,(xi.shape[2],xi.shape[3])))),-1)[0].permute(1,2,0).cpu().detach().numpy())
        x = self.conv0(xi) # 64 * h
        x = torch.cat([x, sty_fea['relu1_1'].detach()], dim=1) # 128 * h
        x = self.conv1(x)
        x = torch.cat([x, sty_fea['relu2_1'].detach()], dim=1) # 256 * h/2
        x = self.conv2(x)
        x = torch.cat([x, sty_fea['relu3_1'].detach()], dim=1) # 512 * h/4
        tx2 = self.enc2(x)
        # tx2 = x
        if flow is not None:
            tx2 = util.bilinear_warp(tx2, self.flow2img(tx2,flow))
        mx2 = self.mask2(tx2)
        # import cv2;cv2.imwrite('oo1.png',254 * sem_mask.type(torch.float)[0][0].cpu().numpy())
        # import cv2;cv2.imwrite('oo2s.png',254 * torch.cat((util.bilinear_warp(sem_mask.type(torch.float), flow),F.interpolate(mx,sem_mask.shape[2:])),-1)[0][0].detach().cpu().numpy())
        return tx2, mx2


class Decoder2(BaseNetwork):
    def __init__(self, input_dim, output_dim=3,n_upsample =2, norm = 'layer',n_res = 2):
        super(Decoder2, self).__init__()
        self.n_res=n_res
        norm_layer =base_function.get_norm_layer('instance')
        self.model = []
        if n_res>0:
            self.model += [base_function.ResBlocks(n_res,input_dim,norm_layer=norm_layer)]
        dim = input_dim
        for i in range(n_upsample):
            self.model+=[
                nn.Upsample(scale_factor=2, mode='bilinear'),
                base_function.ResBlocks(n_res,dim,norm_layer=norm_layer),
                Conv2dBlock(dim, dim//2, 5, 1, 2, norm='layer', activation='LeakyReLU', pad_type='reflect')    
                ]
            dim = dim //2
        self.model = nn.Sequential(*self.model)
        self.final_conv = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')
    
    def forward(self, x, detach_grad:bool = False):
        # x L * h/4
        if detach_grad:
            with torch.no_grad():
                x = self.model(x)
                out = self.final_conv(x)
        else:
            x = self.model(x)
            out = self.final_conv(x)
        return out

#---------------------------------------------------------------------------------------

class Generator2(BaseNetwork):
    def __init__(self,image_nc=3, structure_nc=18, norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False):
        super(Generator2, self).__init__()
        self.pose_encoder = PoseEncoder2(input_dim=structure_nc,dim=64, max_dim=256, n_res=2,norm='instance')
        # self.body_gen = PartialStyleBlock(256)
        # self.garment_gen = PartialStyleBlock(256)
        self.body_gen = StyleBlock(256)
        self.garment_gen = StyleBlock(256)
        
        
    def soft_mask_difference(self,soft_mask_list, mask, seg):
        _,_,h,w = mask.shape
        soft_mask_list.append([mask,F.interpolate(seg.float(),(h,w))])


    def forward(self,vgg, segment_encoder: Encoder2,
                image_decoder: Decoder2,
                flow,
                source_Img, source_Pose, source_Seg, target_Pose, target_Seg):
        from data.dior2_dataset import SEG

        bg_Seg = source_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        target_bg_mask = target_Seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        face_seg = source_Seg[:,SEG.FACE,...].unsqueeze(1)
        target_face_mask = target_Seg[:,SEG.FACE,...].unsqueeze(1)
        skin_seg = (source_Seg[:,SEG.FACE,...] | source_Seg[:,SEG.ARMS,...]| source_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        target_skin_mask = (target_Seg[:,SEG.FACE,...] | target_Seg[:,SEG.ARMS,...]| target_Seg[:,SEG.LEGS,...]).unsqueeze(1)
        
        soft_mask_list=[]
        intermediate_images = []

        bg_texture, bg_mask = segment_encoder(vgg,source_Img, bg_Seg, flow)
        face_texture, face_mask = segment_encoder(vgg,source_Img, face_seg, flow)
        skin_texture, skin_mask = segment_encoder(vgg,source_Img, skin_seg, flow)
        hands_texture, hands_mask = segment_encoder(vgg,source_Img, source_Seg[:,SEG.ARMS,...].unsqueeze(1), flow)
        
        self.soft_mask_difference(soft_mask_list, bg_mask, target_bg_mask)
        self.soft_mask_difference(soft_mask_list, face_mask, target_face_mask)
        self.soft_mask_difference(soft_mask_list, skin_mask, target_skin_mask)

        batch,channels,h,w = skin_texture.shape
        # avg vector over mask region
        skin_area = torch.sum(skin_mask.view(batch, 1, -1), dim=2)
        skin_avg = ((torch.sum((skin_texture * skin_mask).view(batch, channels, -1)
            , dim=2) / (skin_area+1))).unsqueeze(-1).unsqueeze(-1)
        # skin_avg = torch.nn.AdaptiveAvgPool2d(1)(skin_texture * skin_mask)
        

        garments_ids = [SEG.HAIR, SEG.SHOES, SEG.PANTS, SEG.UPPER, SEG.HAT]
        garments_textures_and_masks = {}
        fg_mask = skin_mask
        for index,seg_id in enumerate(garments_ids):
            seg = source_Seg[:,seg_id,...].unsqueeze(1)
            t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
            garment_texture, garment_mask = segment_encoder(vgg,source_Img, seg, flow)
            self.soft_mask_difference(soft_mask_list, garment_mask, t_seg)
            garments_textures_and_masks[seg_id]=[garment_texture, garment_mask]
            fg_mask = torch.min(torch.ones_like(fg_mask),torch.max(fg_mask,garment_mask))  
            fg_mask = torch.max(fg_mask,garment_mask)     

        # body_texture = (1 - bg_mask) * skin_avg +  bg_mask * bg_texture + face_mask *  face_texture
        # bg_mask = 1-fg_mask
        body_texture = fg_mask * skin_avg +  (1-fg_mask) * bg_texture + face_mask *  face_texture + hands_mask* hands_texture
        # imsave import cv2;cv2.imwrite('obody.png',254 * fg_mask[0].detach().cpu().numpy())


        z_pose = self.pose_encoder(target_Pose)
        z_body = self.body_gen(z_pose, body_texture)
        #TODO ad FACE texture

        body_img = image_decoder(z_body[:1], detach_grad=True).cpu()
        intermediate_images.append(body_img)
        # imsave import cv2;cv2.imwrite('obody.png',127 + 128 * body_img[0].detach().permute(1,2,0).cpu().numpy())
        
        z_style = z_body
        for index,seg_id in enumerate(garments_ids):
            garment_texture, garment_mask = garments_textures_and_masks[seg_id]
            z_style = self.garment_gen(z_style, garment_texture, garment_mask)
            # for i in range(len(z_style_next)):
            #         z_style[i] = z_style_next[i] * garment_mask[i] + z_style[i] * (1 - garment_mask[i])
            if index+1==len(garments_ids):
                final_img = image_decoder(z_style)
            else:
                with torch.no_grad():
                    garment_img = image_decoder(z_style[:1], detach_grad=True).cpu()
                    intermediate_images.append(garment_img)     
        
        # imsave import cv2;cv2.imwrite('ofinal.png',127 + 128 * final_img[0].detach().permute(1,2,0).cpu().numpy())
        return final_img, soft_mask_list, intermediate_images

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel, affine=False, track_running_stats=False)
        
    def forward(self, input, gamma, beta,mask=None):
        if not isinstance(mask, torch.Tensor): # mask == None:
            out = self.norm(input)
            out = gamma * out + beta
            return out   
        else:
            out = self.norm(input)
            out = gamma * out + beta
            return out * mask + input * (1 - mask)

class StyleBlock(nn.Module):
    '''
    https://github.com/cuiaiyu/utilities/blob/63e693c774990791482b565d030142f0071a14f7/models/base_networks.py#L231
    '''
    def __init__(self, out_nc=None, relu_type='relu'):
        super(StyleBlock, self).__init__()
        if relu_type == 'relu':
            self.relu = nn.ReLU(True) 
        elif relu_type == 'leakyrelu' :
            self.relu =  nn.LeakyReLU(0.2, True)
        else:
            self.relu = nn.ReLU6(True)
        self.ad_norm = AdaptiveInstanceNorm(out_nc)
        self.norm = nn.InstanceNorm2d(out_nc, affine=False, track_running_stats=False)
        latent_nc = out_nc
        
        self.conv1 = nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        
        
    def forward(self, x, style, mask=None, cut=False, adain=True):
        if len(style.size()) == 2:
            style = style[:,:, None, None]
        gamma, beta = style.chunk(2,1)
        gammas = gamma.chunk(2,1)
        betas = beta.chunk(2,1)
        
        out = x
        out = self.conv1(x)
        if adain:
            out = self.ad_norm(out, gammas[0], betas[0], mask)
        else:
            out = self.norm(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if adain:
            out = self.ad_norm(out, gammas[1], betas[1], mask)
        else:
            out = self.norm(out)
        if cut:
            return out
        return out + x      


class AdaptivePartialInstanceNorm(nn.Module):
    # https://github.com/cuiaiyu/utilities/blob/63e693c774990791482b565d030142f0071a14f7/models/base_networks.py
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        
    def forward_base(self, input, gamma, beta):
        out = self.norm(input)
        out = gamma * out + beta
        return out   

    def forward(self, input, gamma, beta, mask=None):
        if not isinstance(mask, torch.Tensor):
            return self.forward_base(input, gamma, beta)
        # import pdb; pdb.set_trace()

        mask = mask.float()
        N,_,H,W = input.size()
        area = torch.sum(mask.view(N,-1,H*W),-1) + 1e-5

        # compute mean and variance only in the valid region
        valid_img = (input * mask).view(N,-1,H*W)
        mu = torch.sum(valid_img, -1) / area
        sigma = torch.sum((valid_img - mu.unsqueeze(-1)) ** 2, -1) / area
        mu, sigma = mu.unsqueeze(-1).unsqueeze(-1), sigma.unsqueeze(-1).unsqueeze(-1) + 1e-12
        # shift
        out = (input - mu.detach()) / torch.sqrt(sigma).detach()
        out = out * gamma + beta
        return out * mask + input * (1 - mask)


class PartialStyleBlock(nn.Module):
    # https://github.com/cuiaiyu/utilities/blob/63e693c774990791482b565d030142f0071a14f7/models/base_networks.py
    def __init__(self, out_nc=None, relu_type='relu'):
        super(PartialStyleBlock, self).__init__()
        if relu_type == 'relu':
            self.relu = nn.ReLU(True) 
        elif relu_type == 'leakyrelu' :
            self.relu =  nn.LeakyReLU(0.2, True)
        else:
            self.relu = nn.ReLU6(True)
        self.ad_norm = AdaptivePartialInstanceNorm(out_nc)
        latent_nc = out_nc
        
        self.conv1 = nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        
        
    def forward(self, x, style, mask=None):
        if len(style.size()) == 2:
            style = style[:,:, None, None]
        gamma, beta = style.chunk(2,1)
        gammas = gamma.chunk(2,1)
        betas = beta.chunk(2,1)
        
        out = x
        out = self.conv1(x)
        out = self.ad_norm(out, gammas[0], betas[0], mask)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ad_norm(out, gammas[1], betas[1], mask)
        
        # out = self.relu(out)
        return out + x
