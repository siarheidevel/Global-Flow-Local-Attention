import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/deeplab/devel/Global-Flow-Local-Attention')

# from model.base_model import BaseModel
from model.networks import base_function,external_function2

# from model.networks import base_function, external_function, BaseNetwork
# import model.networks as network
from util import task, util
# import itertools
# import data as Dataset
# import numpy as np
# from itertools import islice
# import random
# import os



class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='ReLU', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.pad = base_function.get_padding(pad_type, padding)()
        kwargs_conv = {'kernel_size': kernel_size, 'stride': stride}
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


class MaskedAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        N,C,H,W = x.size()
        mask = F.interpolate(mask, (H,W))
        x = x.view(N,C,-1).sum(-1)
        area = mask.view(N,1,-1).sum(-1) + 1
        return (x / area)[:,:,None,None]


class ADGANOriginalEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_downsample=3, norm_type='none', relu_type='ReLU'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(ADGANOriginalEncoder, self).__init__()
        # self.vgg_listen_list = ['conv_1_2', 'conv_2_2', 'conv_3_2', 'conv_4_2']
        # self.vgg_listen_list = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1']
        self.vgg_listen_list = ['relu1_1','relu2_1','relu3_1','relu4_1']
        model = []
        model += [Conv2dBlock(input_nc, ngf, 7, 1, 3, norm_type, relu_type)]
        vgg_ngf = 64
        # n_downsample = 2
        for i in range(n_downsample):  # add downsampling layers
            mult = 2 ** i
            curr_ngf = ngf * mult
            curr_vgg = vgg_ngf * mult
            model += [Conv2dBlock(curr_ngf + curr_vgg, curr_ngf * 2, 3, 2, 1, norm_type, relu_type)]
        
        self.model = nn.Sequential(*model)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.maskpool = MaskedAvgPool()
        self.conv_last = nn.Conv2d((curr_ngf + curr_vgg) * 2, output_nc, 1)
        self.reduced = False
   
    
    def forward(self, x, vgg, fmap_only=False, mask=None):
        """Standard forward"""
        v_layers = self.vgg_listen_list
        with torch.no_grad():
            vgg_out = vgg(x)
        retmaps = []
        # import pdb; pdb.set_trace()
        out = self.model[0](x)
        out = self.model[1](torch.cat([out, vgg_out[v_layers[0]]], dim=1)) # 128
        out = self.model[2](torch.cat([out, vgg_out[v_layers[1]]], dim=1)) # 256
        retmaps.append(out)
        if self.reduced or fmap_only:
            return retmaps, None
        out = self.model[3](torch.cat([out, vgg_out[v_layers[2]]], dim=1)) # 512
        retmaps.append(out)
        if isinstance(mask, torch.Tensor):
            out = self.maskpool(torch.cat([out, vgg_out[v_layers[3]]], dim=1), mask)
        else:

            out = self.pool(torch.cat([out, vgg_out[v_layers[3]]], dim=1))
        #out = self.conv_last(out)
        #N = out.size(0)
        #attr_maps = F.interpolate(attr_maps, (64, 44))
        return retmaps, out #out.view(N, -1)

    def final_conv(self, x):
        if self.reduced or not isinstance(x, torch.Tensor):
            return None
        out = self.conv_last(x)
        N = out.size(0)
        return out.view(N, -1)


class AiyuDecoder(nn.Module):
    def __init__(self, n_upsample, dim, output_dim, activ='LeakyReLU', pad_type='zero'):
        super(AiyuDecoder, self).__init__()
        norm_layer = base_function.get_norm_layer('instance')
        nonlinearity = base_function.get_nonlinearity_layer(activation_type=activ)
        self.model = []
        # upsampling blocks
        self.model += [base_function.ResBlocks(2, dim, norm_layer=norm_layer, nonlinearity=nonlinearity)]
        # activ = 'lrelu' if activ == 'leakyrelu' else activ
        for i in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                base_function.ResBlocks(2, dim, norm_layer=norm_layer, nonlinearity=nonlinearity),
                # 'layer, instance
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='layer', activation=activ, pad_type=pad_type)
                ]
                        
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        #return x
        return self.model(x)



##------------------------------------------------------------------------------------

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


class AdaptivePartialInstanceNorm(nn.Module):
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

    
class StyleBlock(nn.Module):
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


if __name__ == '__main__':
    vgg = external_function2.VGG19Limited()
    encoder = ADGANOriginalEncoder(3,256,n_downsample=3)
    img = torch.randn(1,3,256,192)
    out = encoder(img, vgg)
    print(out)






######################################################################################################################################
#Multi encoders full model
class PoseEncoder2(BaseNetwork):
    def __init__(self, input_dim, dim =64, max_dim = 256, n_res = 5, norm='instance', 
                 activation='LeakyReLU', pad_type = 'reflect'):
        super(PoseEncoder2, self).__init__()
        norm_layer = base_function.get_norm_layer(norm)
        nonlinearity = base_function.get_nonlinearity_layer(activation_type=activation)
        max_inner_dim = 128
        self.conv0 = nn.Conv2d(input_dim, dim, 5, 1, 2, padding_mode='zeros')
        next_dim = min( max_inner_dim, dim * 2)
        self.enc0 = base_function.EncoderBlock(dim, next_dim, norm_layer=None, nonlinearity=nonlinearity) # 128 h/2
        dim = next_dim
        next_dim = min( max_inner_dim, dim * 2)
        self.enc1 = base_function.EncoderBlock(dim, next_dim, norm_layer=norm_layer, nonlinearity=nonlinearity) # 256 h/4       
        self.n_res = n_res
        if self.n_res>0:
            self.res = nn.Sequential(*[ResBlockDilated(next_dim,norm_layer = norm_layer,dilation=2) for i in range(self.n_res)])
            # base_function.ResBlocks(num_blocks=self.n_res, input_nc = next_dim,
            #     norm_layer = norm_layer)
        self.res_final = base_function.ResBlock(128,max_dim,norm_layer=norm_layer, nonlinearity=nonlinearity)  

    def forward(self, x):
        # x  18 * h
        x = self.conv0(x) # 64 * h/2
        x = self.enc0(x) # 128 * h/2
        x = self.enc1(x) # 128 h/4
        if self.n_res>0:
            x = self.res(x) #128 h/4      
        x = self.res_final(x)
        return x


class SoftShapeMask(BaseNetwork):
    def __init__(self, dim):
        super(SoftShapeMask, self).__init__()
        self.conv1 = Conv2dBlock(dim, dim//2 , 3, 1, 1, norm='none',activation='LeakyReLU', pad_type='zero')
        self.conv2 = Conv2dBlock(dim//2, 1 , 3, 1, 1, norm='none',activation='none', pad_type='zero')
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x) 
        # torch.mean(x),torch.std(x),torch.min(x),torch.max(x)
        # return torch.clamp(x,0,1)
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
        dim = dim * 2
        self.enc2 = Conv2dBlock(dim, dim//2, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type) # 512 -> 256
        self.mask2 = SoftShapeMask(dim//2)

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
        x = self.conv1(x)
        x = torch.cat([x, sty_fea['relu2_1'].detach()], dim=1) # 256 * h/2
        x = self.conv2(x)
        x = torch.cat([x, sty_fea['relu3_1'].detach()], dim=1) # 512 * h/4
        tx2 = self.enc2(x)
        # tx2 = x
        tx2 = util.bilinear_warp(tx2, self.flow2img(tx2,flow))
        mx2 = self.mask2(tx2)
        # import cv2;cv2.imwrite('oo1.png',254 * sem_mask.type(torch.float)[0][0].cpu().numpy())
        # import cv2;cv2.imwrite('oo2s.png',254 * torch.cat((util.bilinear_warp(sem_mask.type(torch.float), flow),F.interpolate(mx,sem_mask.shape[2:])),-1)[0][0].detach().cpu().numpy())
        return tx2, mx2

class Decoder2(BaseNetwork):
    def __init__(self, input_dim, output_dim, norm = 'layer'):
        super(Decoder2, self).__init__()
        self.model = []
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = Conv2dBlock(input_dim, input_dim//2, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')
        self.conv1 = Conv2dBlock(input_dim//2, input_dim//2, 5, 1, 2, norm=norm, activation='LeakyReLU', pad_type='reflect')
        self.final_conv = Conv2dBlock(input_dim//2, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')
    
    def forward(self, x, detach_grad:bool = False):
        # x L * h/4
        if detach_grad:
            x = x.detach()
        # TODO layerNorm + activation
        x = self.up(x) # L * h/2
        x = self.conv2(x) # L * h/2

        x = self.up(x) # L/2 * h
        x = self.conv1(x) # L/2 * h
        out = self.final_conv(x) # 3 * h
        if detach_grad:
            out = out.detach()
        return out

class Style2(BaseNetwork):
    def __init__(self, style_dim=256, label_nc = 512, spade_config_str ='spadeinstance3x3',use_spectral= False, learned_shortcut =True):
        super(Style2, self).__init__()
        self.sp2 = base_function.SPADEResnetBlock(input_nc = style_dim, output_nc = style_dim,
                                                          hidden_nc = style_dim, label_nc =label_nc,
                                                          spade_config_str=spade_config_str,use_spect=use_spectral,
                                                          learned_shortcut=learned_shortcut)
        
    def forward(self, prev_state, texture, mask):
        state = prev_state # L * H/4
        tx = texture # 512 * H/4
        m = mask # 1 * H/4
        style = state * (1 -m) + self.sp2(state, tx) * m
        return style


class Generator2(BaseNetwork):
    def __init__(self,image_nc=3, structure_nc=18, norm='instance', activation='LeakyReLU',
            use_spect=False, use_coord=False):
        super(Generator2, self).__init__()
        self.pose_encoder = PoseEncoder2(input_dim=18,dim=64, max_dim=256, n_res=6,norm='instance')
        self.style_gen = Style2(style_dim = 256, label_nc = 256, spade_config_str='spadeinstance3x3', use_spectral=use_spect, learned_shortcut=True)    
    
    
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
            # body_texture = (1 - bg_masks[i]) * skin_avg +  bg_masks[i] * bg_textures[i] + skin_textures[i] *  skin_textures[i]
            body_textures.append(body_texture)
            body_masks.append(torch.ones_like(body_texture))
        return  body_textures, body_masks


    def forward(self, segment_encoder: Encoder2,
                image_decoder: Decoder2,
                flow,
                source_Img, source_Pose, source_Seg, target_Pose, target_Seg):
        from data.dior2_dataset import SEG


        # flow_fields = flow_generator(source_Img, source_Pose, target_Pose)[0]
        # # take h/4, w/4 flow                
        # flow = flow_fields[-1]
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
        bg_texture, bg_mask = segment_encoder(source_Img, bg_Seg, flow)
        face_texture, face_mask = segment_encoder(source_Img, face_seg, flow)
        skin_texture, skin_mask = segment_encoder(source_Img, skin_seg, flow)
        Generator2.soft_mask_difference(soft_mask_list, [bg_mask], target_bg_mask)
        Generator2.soft_mask_difference(soft_mask_list, [face_mask], target_face_mask)
        Generator2.soft_mask_difference(soft_mask_list, [skin_mask], target_skin_mask)
        
        
        body_textures, body_masks=self.create_body_texture([bg_texture],[bg_mask], [face_texture],[face_mask], [skin_texture],[skin_mask])

        z_pose = self.pose_encoder(target_Pose)
        z_style = self.style_gen(z_pose, body_textures[0], body_masks[0])
        #TODO ad FACE texture

        body_img = image_decoder(z_style, detach_grad=True)
        intermediate_images.append(body_img.cpu())
        # imsave import cv2;cv2.imwrite('obody.png',127 + 128 * body_img[0].detach().permute(1,2,0).cpu().numpy())

        garments = [SEG.HAIR, SEG.SHOES, SEG.PANTS, SEG.UPPER, SEG.HAT]
        
        for index,seg_id in enumerate(garments):
            seg = source_Seg[:,seg_id,...].unsqueeze(1)
            t_seg = target_Seg[:,seg_id,...].unsqueeze(1)
            garment_textures, garment_masks = segment_encoder(source_Img, seg, flow)
            Generator2.soft_mask_difference(soft_mask_list, [garment_masks], t_seg)
            z_style = self.style_gen(z_style, garment_textures, garment_masks)
            # for i in range(len(z_style_next)):
            #         z_style[i] = z_style_next[i] * garment_mask[i] + z_style[i] * (1 - garment_mask[i])
            if index+1==len(garments):
                final_img = image_decoder(z_style)
            else:
                with torch.no_grad():
                    garment_img = image_decoder(z_style, detach_grad=True)
                    intermediate_images.append(garment_img.cpu())     
        
        # imsave import cv2;cv2.imwrite('ofinal.png',127 + 128 * final_img[0].detach().permute(1,2,0).cpu().numpy())
        return final_img, soft_mask_list, intermediate_images



