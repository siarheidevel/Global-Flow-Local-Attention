import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function, BaseNetwork
from model.networks.base_network import freeze_network, init_network, print_network
from data.dior_dataset import SEG
import model.networks as network
import model.networks.dior_models as dior_models
from util import task, util
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

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        # parser.set_defaults(save_input=False)
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['correctness_gen', 'regularization_flow']

        self.visual_names = ['input_P1','input_P2','input_BP1', 'input_BP2','img_gen','flow_fields'
            ]
        self.model_names = ['FlowG']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        


        # define the generator
        # self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
        #                               layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
        #                               norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        # network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18, ngf=32, img_f=256,
        #                 encoder_layer=4, attn_layer=opt.attn_layer,
        #                 norm='instance', activation='LeakyReLU',
        #                 use_spect=opt.use_spect_g, use_coord=False)(torch.randn((1,3,256,176)),torch.randn((1,18,256,176)),torch.randn((1,18,256,176)))
        self.net_FlowG = init_network(network.generator.PoseFlowNetGenerator(image_nc=3, structure_nc=18, ngf=64, img_f=512,
                        encoder_layer=4, attn_layer=opt.attn_layer,
                        norm='instance', activation='LeakyReLU',
                        use_spect=opt.use_spect_g, use_coord=False), opt)
        self.flow2color = util.flow2color()

        if self.isTrain:
            # define the loss functions
            # geo loss
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)

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

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)

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
        self.flow_fields, _ = self.net_FlowG(self.input_P1, self.input_BP1, self.input_BP2)
        self.img_gen = [util.bilinear_warp(self.input_P1,self.flow_fields[0]),
            util.bilinear_warp(self.input_P1,self.flow_fields[1]),
            util.bilinear_warp(self.input_P1,self.flow_fields[2])]



    
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