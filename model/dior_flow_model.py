import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function2, BaseNetwork
from model.networks.base_network import freeze_network, init_network, print_network
from data.dior_dataset import SEG
import model.networks as network
import model.networks.dior_models as dior_models
from util import task, util, openpose_utils
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os

####################################################################################################################

class Dior_Flow(BaseModel):
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
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1 parameter')

        
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for the affine regularization loss')

        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        # parser.set_defaults(save_input=False)
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['correctness_gen', 'regularization_flow', 'l1', 'content_gen']
        self.loss_names = ['correctness_gen', 'regularization_flow']

        self.visual_names = ['input_P1','input_P2','input_BP1', 'input_BP2','img_gen','flow_fields'
            ,'input_BP1_RGB','input_BP2_RGB','input_Mask1','input_Mask2']
        self.model_names = ['FlowG']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        

        self.vgg = external_function2.VGG19Limited('./saved_models/vgg19-dcbb9e9d.pth').to(opt.device)
        # define the generator
        # self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
        #                               layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
        #                               norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        # network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18, ngf=32, img_f=256,
        #                 encoder_layer=4, attn_layer=opt.attn_layer,
        #                 norm='instance', activation='LeakyReLU',
        #                 use_spect=opt.use_spect_g, use_coord=False)(torch.randn((1,3,256,176)),torch.randn((1,18,256,176)),torch.randn((1,18,256,176)))
        self.net_FlowG = init_network(network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18+3, ngf=64, img_f=512,
                        encoder_layer=5, attn_layer=opt.attn_layer,
                        norm='instance', activation='LeakyReLU',
                        use_spect=opt.use_spect_g, use_coord=False), opt)
        self.flow2color = util.flow2color()

        if self.isTrain:
            # define the loss functions
            # geo loss
            self.Correctness = external_function2.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function2.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)

            self.L1loss = torch.nn.L1Loss()

            self.Perceptual = external_function2.PerceptualLoss().to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_FlowG.parameters())),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            
        # load the pre-trained model and schedulers
        self.setup(opt)
        print('---------- Networks initialized -------------')
        print_network(self.net_FlowG)
        print('-----------------------------------------------')


    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1, input_BP1, input_SP1, input_M1 = input['P1'], input['BP1'], input['SP1'], input['M1']
        input_P2, input_BP2, input_SP2 = input['P2'], input['BP2'], input['SP2']
        input_BP1_RGB, input_BP2_RGB = input['BP1_RGB'], input['BP2_RGB']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1_RGB = input_BP1_RGB.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2_RGB = input_BP2_RGB.cuda(self.gpu_ids[0], non_blocking=True)
            # stiching all no background layes
            input_SP1=torch.max(input_SP1[:,1:,...],dim=1)[0].unsqueeze(1).float()
            input_SP2=torch.max(input_SP2[:,1:,...],dim=1)[0].unsqueeze(1).float()
            self.input_Mask1 = input_SP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_Mask2 = input_SP2.cuda(self.gpu_ids[0], non_blocking=True)
            

        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])                     
                

    def forward(self):
        """Run forward processing to get the inputs"""
        # self.flow_fields, _ = self.net_FlowG(self.input_P1, self.input_BP1, self.input_BP2)
        self.flow_fields, _ = self.net_FlowG(
            self.input_P1, torch.cat((self.input_BP1_RGB,self.input_BP1),1),
            torch.cat((self.input_BP2_RGB,self.input_BP2),1))
        # self.flow_fields, _ = self.net_FlowG(
        #     self.input_P1, torch.cat((self.input_Mask1,self.input_BP1),1),
        #     torch.cat((self.input_Mask2,self.input_BP2),1))
        with torch.no_grad():
            self.img_gen = [util.bilinear_warp(self.input_P1,self.flow_fields[0]),
                util.bilinear_warp(self.input_P1,self.flow_fields[1]),
                util.bilinear_warp(self.input_P1,4*F.interpolate(self.flow_fields[1],self.input_P2.shape[2:], mode='bilinear')),
                ]
        # self.img_gen = [util.bilinear_warp(self.input_P1,self.flow_fields[0]),
        #     util.bilinear_warp(self.input_P1,self.flow_fields[1]),
        #     util.bilinear_warp(self.input_P1,self.flow_fields[2])]



    def backward_G(self):
        """Calculate training loss for the generator"""
        vgg_features_p1 = self.vgg(self.input_P1, max_layer='relu4_1')
        vgg_features_p2 = self.vgg(self.input_P2, max_layer='relu4_1')
        # calculate geometric loss #####
        # Calculate Sampling Correctness Loss
        loss_correctness_gen = self.Correctness(vgg_features_p2, vgg_features_p1, self.flow_fields, self.opt.attn_layer)
        self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct

        # Calculate regularization term
        loss_regularization_flow = self.Regularization(self.flow_fields)
        self.loss_regularization_flow = loss_regularization_flow * self.opt.lambda_regularization

        # #l1 loss
        # total_l1_loss = 0
        # total_content_loss = 0
        # for i in range(len(self.flow_fields)):
        #     flow = F.interpolate(self.flow_fields[i],self.input_P2.shape[2:], mode='bilinear')
        #     warped_img = util.bilinear_warp(self.input_P1,flow)
        #     real_img =self.input_P2
        #     vgg_features_warped_img, vgg_features_real_img =self.vgg(warped_img), vgg_features_p2
        #     # warped_img = self.img_gen[i]
        #     # real_img = F.interpolate(self.input_P2, warped_img.shape[2:])
        #     # vgg_features_warped_img, vgg_features_real_img =self.vgg(warped_img), self.vgg(real_img)
        #     total_l1_loss +=  self.L1loss(warped_img, real_img)
        #     loss_content_gen = self.Perceptual(vgg_features_warped_img, vgg_features_real_img)
        #     total_content_loss += loss_content_gen
        # self.loss_l1 = total_l1_loss * self.opt.lambda_rec
        # self.loss_content_gen = loss_content_gen*self.opt.lambda_content
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