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
    '''
        pad -> conv -> norm -> activation
    '''
    def __init__(self, input_dim ,output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='ReLU', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.pad = base_function.get_padding(pad_type, padding)()
        kwargs_conv = {'kernel_size': kernel_size, 'stride': stride, 'dilation': dilation}
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

# pose encoder ---------------------------------------------------------------------------------------------
class PoseEncoder(BaseNetwork):
    def __init__(self, input_dim, n_layers=4, dim =64, max_dim = 256, n_res = 5, norm='instance', 
                 activation='LeakyReLU', pad_type = 'reflect'):
        super(PoseEncoder, self).__init__()
        norm_layer = base_function.get_norm_layer(norm)
        nonlinearity = base_function.get_nonlinearity_layer(activation_type=activation)
        self.n_layers = n_layers
        self.n_res = n_res
        self.conv0 = nn.Sequential(
                nn.Conv2d(input_dim, dim, 5, 1, 2, padding_mode='zeros'),
                nonlinearity)
        for i in range(self.n_layers):
            next_dim = min(max_dim, dim * 2)
            enc = nn.Sequential(Conv2dBlock(dim, next_dim,4 ,2, 1, norm),
                                Conv2dBlock(next_dim, next_dim, 3, 1, 1, norm))
            setattr(self,'enc'+str(i), enc)
            dim = next_dim
        
        if self.n_res > 0:
            self.res = base_function.ResBlocks(n_res,dim, norm_layer=norm_layer, nonlinearity=nonlinearity)
    
    def forward(self, x):
        # x  18 * h
        x = self.conv0(x) # 64 * h
        for i in range(self.n_layers):
            enc = getattr(self, 'enc'+str(i))
            x = enc(x)
        
        if self.n_res>0:
            x = self.res(x) # max_dim h/2^n_layers
        return x

# image seg encoder -------------------------------------------------------------------
class ImageEncoder(BaseNetwork):
    def __init__(self, input_img_dim, n_layers,k_classes, dim=64,  norm='none', activation='LeakyReLU', pad_type ='reflect'):
        super(ImageEncoder, self).__init__()
        self.n_layers = n_layers
        self.vgg_layers=['relu1_1','relu2_1','relu3_1','relu4_1']
        self.conv = Conv2dBlock(input_img_dim, dim, 7, 1, 3, norm=norm,activation=activation, pad_type=pad_type)# 3->64,concat
        for i in range(self.n_layers):
            next_dim = dim *2
            enc = Conv2dBlock(next_dim,next_dim,4,2,1, norm, activation)
            enc_out = Conv2dBlock(next_dim*2, dim,3,1,1, norm, activation)
            setattr(self,'enc'+str(i), enc)
            setattr(self,'enc_out'+str(i), enc_out)
            setattr(self,'gamma_beta_'+str(i),
                nn.Sequential(
                    Conv2dBlock(k_classes*dim, 128, 3,1,1, activation=activation),
                    nn.Conv2d(128, dim*2, 3,1,1)
                    )
                )
            dim = next_dim
    
    def forward(self, img, seg, vgg_network):
        #num classes
        n_class = seg.size(1)
        layers = {}
        gammas = {}
        betas = {}
        for k in range(n_class):
            x = img * seg[:,k,:,:].unsqueeze(1) ## import cv2;cv2.imwrite('obody.png',127 + 128 * x[0].detach().permute(1,2,0).cpu().numpy())
            vgg_features = vgg_network(x)
            x = torch.cat([self.conv(x), vgg_features[self.vgg_layers[0]].detach()], dim=1)
            for i in range(self.n_layers):
                enc = getattr(self, 'enc'+ str(i))
                x = torch.cat([enc(x), vgg_features[self.vgg_layers[i+1]].detach()], dim=1)
                enc_out = getattr(self, 'enc_out'+ str(i))             
                if i not in layers:
                    layers[i] = enc_out(x)
                else:
                    layers[i] = torch.cat([layers[i],enc_out(x)], dim=1)

        # calculate gamma and beta 
        for i in range(self.n_layers):
            x = layers[i]
            gamma_beta = getattr(self, 'gamma_beta_'+ str(i))(x)
            gamma = gamma_beta[:,:gamma_beta.shape[1]//2,:,:]
            beta = gamma_beta[:,gamma_beta.shape[1]//2:,:,:]
            gammas[i]=gamma
            betas[i]=beta
        return layers, gammas, betas

# STyle transfer module --------------------------------------------------------------------------------------------------
class SAWN(nn.Module):
    def __init__(self, dim, norm ='instance'):
        super(SAWN, self).__init__()
        # kwargs_conv = {'kernel_size': kernel_size, 'stride': 1, 'dilation': 1, 'padding': padding}
        if norm == 'instance':
            self.normalization = nn.InstanceNorm2d(dim, affine=False)
        elif norm == 'batch':
            self.normalization = nn.BatchNorm2d(dim, affine=False)
        # self.gamma = base_function.coord_conv(n_class*dim, dim,use_spect=spectral,**kwargs_conv)
        # self.beta = base_function.coord_conv(n_class*dim, dim,use_spect=spectral,**kwargs_conv)


    def forward(self, h, gamma, beta, flow, flow_mask):
        h_norm = self.normalization(h)
        # g = self.gamma(segmentations)
        # b =self.beta(segmentations)
        gamma_w = util.bilinear_warp(gamma, flow)
        beta_w =util.bilinear_warp(beta, flow)
        h_new = (gamma_w * flow_mask + h * (1-flow_mask)) * h_norm + beta_w
        return h_new


class SAWNResBlock(nn.Module):
    def __init__(self, fin, fout,norm ='instance', kernel_size=3, padding=1, spectral=False):
        super(SAWNResBlock,self).__init__()
        self.learned_shortcut = (fin != fout)
        # create conv layers
        kwargs_conv = {'kernel_size': 3, 'stride': 1, 'dilation': 1, 'padding': 1}
        self.actvn = nn.LeakyReLU()
        self.conv_0 = base_function.coord_conv(fin, fin, **kwargs_conv)
        self.conv_1 = base_function.coord_conv(fin, fout, **kwargs_conv)
        if self.learned_shortcut:
            kwargs_conv = {'kernel_size': 3, 'stride': 1, 'dilation': 1, 'padding': 1, 'bias': False}
            self.conv_s = base_function.coord_conv(fin, fout, **kwargs_conv)
        
        
        self.sawn_0= SAWN(fin,norm)
        self.sawn_1= SAWN(fin,norm)
        if self.learned_shortcut:
            self.sawn_s= SAWN(fin,norm)
    
    def forward(self, h, gamma, beta, flow, flow_mask):
        h_s = h
        if self.learned_shortcut:
           h_s =  self.conv_s(self.actvn(self.sawn_s(h,gamma,beta,flow,flow_mask)))
        d_h = self.conv_0(self.actvn(self.sawn_0(h,gamma,beta,flow,flow_mask)))
        d_h = self.conv_1(self.actvn(self.sawn_1(d_h,gamma,beta,flow,flow_mask)))
        return d_h + h_s


class Generator(BaseNetwork):
    def __init__(self, n_layers, dim=32):
        super(Generator, self).__init__()
        self.n_layers = n_layers
        self.pose_encoder = PoseEncoder(input_dim=18, n_layers=n_layers,dim=dim, max_dim=1024, n_res=2)
        self.conv_img = nn.Conv2d(dim, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        for i in range(n_layers):
            next_dim = dim * 2
            sawn_block = SAWNResBlock(next_dim, dim)
            setattr(self, 'block'+str(i), sawn_block)
            dim = next_dim
        
        
    
    def forward(self, image_source, pose_source, seg_source, pose_target, flow_model, image_encoder:ImageEncoder, vgg_model):
        hidden = self.pose_encoder(pose_target)
        flows,flow_masks = flow_model(image_source, pose_source, pose_target)
        layers, gammas, betas = image_encoder(image_source, seg_source, vgg_model)

        for layer_num in reversed(range(len(layers))):
            gamma = gammas[layer_num]
            beta = betas[layer_num]
            flow = flows[len(flows)-1-layer_num]
            flow_mask =flow_masks[len(flow_masks)-1-layer_num]
            hidden = getattr(self,'block'+str(layer_num))(hidden, gamma, beta, flow, flow_mask)
            hidden = self.up(hidden)
        img_gen = self.conv_img(F.leaky_relu(hidden, 2e-1))
        img_gen = F.tanh(img_gen)
        
        return flows,flow_masks,img_gen
        
                
