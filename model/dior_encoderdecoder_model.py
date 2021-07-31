import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function, BaseNetwork
from data.dior_dataset import SEG
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

class Dior_EncoderDecoder(BaseModel):
    """
       Dior encoder - decoder pretraining with inpainting
    """
    def name(self):
        return "Dress in order Person Image Generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--style_dim', type=int, default=128, help='dimension of style vector')
        parser.add_argument('--style_blocks', type=int, default=7, help='number of style blocks')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1 parameter')

        
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')

        parser.add_argument('--lambda_seg', type=float, default=0.1, help='weight soft mask segmentation')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['l1', 'content_gen', 'style_gen']

        self.visual_names = ['input_P1','input_P2','input_M1','input_BP1', 'input_BP2','input_SP1','input_SP2',
            'img_gen', 'soft_mask_list'
            ]
        self.model_names = ['Enc','Dec']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        


        # define the generator
        # self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
        #                               layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
        #                               norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        # self.net_Enc = init_network(dior_multi_model.MultiEncoder2(input_img_dim=3,style_dim=256, norm='none',activation='LeakyReLU',pad_type='reflect'), opt)
        # self.net_Dec = init_network(dior_multi_model.MultiDecoder2(input_dim=256, output_dim=3,norm='layer'), opt)\
        self.net_Enc = init_network(dior_single_model.Encoder2(input_img_dim=3,style_dim=256, norm='none',activation='LeakyReLU',pad_type='reflect'), opt,init_type='kaiming')
        self.net_Dec = init_network(dior_single_model.Decoder2(input_dim=256, output_dim=3,norm='layer'), opt,init_type='kaiming')
        # check weights torch.sum(next(self.net_Enc.cpu().vgg.parameters())- next(external_function.VGG19().parameters()))
        

        if self.isTrain:
            # define the loss functions
            # content loss:l1 + content + style
            self.L1loss = torch.nn.L1Loss().to(opt.device)
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # # segmentation loss
            # self.SegSoftmaskloss = torch.nn.BCELoss()       
           

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_Enc.parameters()),
                                               filter(lambda p: p.requires_grad, self.net_Dec.parameters())),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            
        # load the pre-trained model and schedulers
        self.setup(opt)
        print('---------- Networks initialized -------------')
        print_network(self.net_Enc)
        print_network(self.net_Dec)
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
        """Run forward processing to get the inputs"""
        self.input_P1 = self.input_P1 * self.input_M1.unsqueeze(1)
        self.input_SP1 = self.input_SP1 * self.input_M1.unsqueeze(1)
        self.soft_mask_list=[]
        b,_,h,w = self.input_P1.size()
        flow0 = torch.zeros((b,2,int(h/4),int(w/4)), device=self.input_P1.device)
        # flow1 = torch.zeros((b,2,int(h/2),int(w/2)), device=self.input_P1.device)
        # TODO soft mask train
        seg_mask =torch.ones((b,1,h,w), device=self.input_P1.device)
        tx, mx = self.net_Enc(seg_img=self.input_P1, seg_mask = seg_mask , flow=flow0)
        # self.soft_mask_difference(tx_mask,seg_mask)
        self.img_gen = self.net_Dec(tx)

    # def forward(self):
    #     """Run forward processing to get the inputs"""
    #     self.input_P1 = self.input_P1 * self.input_M1.unsqueeze(1)
    #     self.input_SP1 = self.input_SP1 * self.input_M1.unsqueeze(1)
    #     self.soft_mask_list=[]
    #     b,_,h,w = self.input_P1.size()
    #     flow = torch.zeros((b,2,int(h/4),int(w/4)), device=self.input_P1.device)
    #     # TODO soft mask train
    #     seg_mask =torch.ones((b,1,h,w), device=self.input_P1.device)
    #     tx,tx_mask = self.net_Enc(seg_img=self.input_P1, seg_mask = seg_mask , flow=flow)
    #     self.soft_mask_difference(tx_mask,seg_mask)
    #     self.img_gen = self.net_Dec(tx)

    def soft_mask_difference(self, mask, seg):
        self.soft_mask_list.append([mask,F.interpolate(seg.float(),mask.shape[2:])])

    def backward_G(self):
        """Calculate training loss for the generator"""
        ########## Content loss #######################
        # Calculate l1 loss
        loss_l1 = self.L1loss(self.img_gen, self.input_P2)
        self.loss_l1 = loss_l1 * self.opt.lambda_rec

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2)  
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content

        # ########## Segmentation mask loss #################
        # loss_mask = 0
        # for soft_mask_target_mask in self.net_G.soft_mask_list:
        #     soft_mask, target_mask = soft_mask_target_mask
        #     loss_mask+=self.SegSoftmaskloss(soft_mask,target_mask)
        # # loss_mask = loss_mask / len(self.net_G.soft_mask_list)
        # self.loss_mask = loss_mask * self.opt.lambda_seg

        
        
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
