import torch
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import os.path as osp
import os
import cv2
import sys
sys.path.append(osp.abspath(osp.dirname(__file__)))
os.chdir(osp.abspath(osp.dirname(__file__)))
CUDA_DEVICE= 'cuda:1'
torch.cuda.set_device(CUDA_DEVICE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.networks import base_function, external_function2, BaseNetwork
from model.networks.base_network import freeze_network, init_network, print_network
import model.networks as network
# import model.networks.dior_models as dior_models
# import model.networks.dior_multi_model as dior_multi_model
import model.networks.dior_single_model2 as dior_single_model2
import data.dior2_dataset as diordataset
from data.dior2_dataset import Dior2Dataset
from util import task, util, pose_utils
from data.dior2_dataset import SEG



class Inferencer:
    def __init__(self, device='cuda') -> None:
        self.image_size = (256,192)
        self.device = device
        self.trans = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    def load_dior_model(self, weights_dir):
        self.vgg = external_function2.VGG19Limited('./saved_models/vgg19-dcbb9e9d.pth').to(self.device)
        self.net_Enc =dior_single_model2.Encoder2(
            input_img_dim=3,style_dim=256, norm='none',activation='LeakyReLU',pad_type='reflect').to(self.device)
        self.net_Dec = dior_single_model2.Decoder2(
            input_dim=256, output_dim=3,norm='layer').to(self.device)
        self.net_DiorG = dior_single_model2.Generator2(image_nc=3, structure_nc=18, 
                norm='instance', activation='LeakyReLU',
                use_spect=False, use_coord=False)
        self.net_FlowG = network.generator.PoseFlowNetGenerator(
            image_nc=3, structure_nc=18, ngf=64, img_f=512,
                            encoder_layer=4, attn_layer=[2,3],
                            norm='instance', activation='LeakyReLU',
                            use_spect=False, use_coord=False).to(self.device)
        print_network(self.net_FlowG)
        print_network(self.net_Dec)
        print_network(self.net_Enc)
        print_network(self.net_DiorG)
        # load model weights
        self.net_Enc.load_state_dict(torch.load(weights_dir+'/latest_net_Enc.pth'))
        self.net_Dec.load_state_dict(torch.load(weights_dir+'/latest_net_Dec.pth'))
        self.net_FlowG.load_state_dict(torch.load(weights_dir+'/latest_net_FlowG.pth'))
        self.net_DiorG.load_state_dict(torch.load(weights_dir+'/latest_net_DiorG.pth'))

    def get_item_data(self,image_file):
        '''
        returns resized img, seg, pose
        '''
        rgb_image = cv2.imread(image_file)[:,:,[2,1,0]]
        seg = np.load( image_file + '.seg.npz')['mask']
        with open(image_file + '.pose2.txt', 'r') as pose_file:
            points_y,points_x,height,width = pose_file.readline().split('\t')
        bones = pose_utils.load_pose_cords_from_strings(points_y,points_x)

        img_height, img_width = rgb_image.shape[0], rgb_image.shape[1]
        fit_height, fit_width = self.image_size[0], self.image_size[1]
        center = img_height * 0.5 + 0.5, img_width * 0.5 + 0.5
        affine_matrix = Dior2Dataset.get_affine_matrix(center=center, fit=(fit_height, fit_width), angle=0, translate=(0,0),
                scale=1, flip=False)

        resized_img = cv2.warpAffine(rgb_image, affine_matrix[:2],(fit_width, fit_height), 
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        SP1 = cv2.warpAffine(seg.transpose(1,2,0), affine_matrix[:2], (fit_width, fit_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,borderValue=0)

        pose  = pose_utils.cords_to_map(bones, (fit_height, fit_width), affine_matrix=affine_matrix, sigma=6)
        pose = np.transpose(pose,(2, 0, 1))

        

        return {'img':self.trans(resized_img), 'seg':torch.Tensor()}

    def change_garment(self, source_img_file, garment_img_file, change_dress_ids:list):
        source = get_item_data(source_img_file)
        garment = get_item_data(garment_img_file)

        source_img, source_seg, source_pose = source['img'],source['seg'],source['pose']
        garment_img, garment_seg, garment_pose = garment['img'],garment['seg'],garment['pose']

        #generate
        from data.dior2_dataset import SEG
        h,w = self.image_size
        no_flow = torch.zeros((1,2,int(h/4),int(w/4)), device=self.device)

        garment_flow = self.net_FlowG(garment_img, garment_pose, source_pose)[0][-1]

        bg_Seg = source_seg[:,SEG.BACKGROUND,...].unsqueeze(1)
        face_seg = source_seg[:,SEG.FACE,...].unsqueeze(1)
        skin_seg = (source_seg[:,SEG.FACE,...] | source_seg[:,SEG.ARMS,...]| source_seg[:,SEG.LEGS,...]).unsqueeze(1)
        
        soft_mask_list=[]
        intermediate_images = []

        bg_texture, bg_mask = self.net_Enc(self.vgg,source_img, bg_Seg, no_flow)
        face_texture, face_mask = self.net_Enc(self.vgg,source_img, face_seg, no_flow)
        skin_texture, skin_mask = self.net_Enc(self.vgg,source_img, skin_seg, no_flow)
        hands_texture, hands_mask = self.net_Enc(self.vgg,source_img, source_seg[:,SEG.ARMS,...].unsqueeze(1), no_flow)
        
        batch,channels,h,w = skin_texture.shape
        # avg vector over mask region
        skin_area = torch.sum(skin_mask.view(batch, 1, -1), dim=2)
        skin_avg = ((torch.sum((skin_texture * skin_mask).view(batch, channels, -1)
            , dim=2) / (skin_area+1))).unsqueeze(-1).unsqueeze(-1)

        body_texture = (1 - bg_mask) * skin_avg +  bg_mask * bg_texture + face_mask *  face_texture + hands_mask* hands_texture

        # body_texture = fg_mask * skin_avg +  (1-fg_mask) * bg_texture + face_mask *  face_texture + hands_mask* hands_texture
        z_pose = self.pose_encoder(source_pose)
        z_body = self.body_gen(z_pose, body_texture)

        garments_ids = [SEG.HAIR, SEG.SHOES, SEG.PANTS, SEG.UPPER, SEG.HAT]

        z_style = z_body
        for index,seg_id in enumerate(garments_ids):
            if seg_id in change_dress_ids:
                garment_texture, garment_mask = self.net_Enc(
                    self.vgg,garment_img, garment_seg, garment_flow)
            else:
                garment_texture, garment_mask = self.net_Enc(
                    self.vgg,source_img, source_seg[:,seg_id,...].unsqueeze(1), no_flow)
            z_style = self.garment_gen(z_style, garment_texture, garment_mask)
        
        final_img = self.net_Dec(z_style)
        return final_img
    

if __name__=='__main__':
    inference = Inferencer(device=CUDA_DEVICE)
    inference.load_dior_model(weights_dir='')

    img = inference.change_garment(source_img_file='',garment_img_file='',change_dress_ids=[SEG.UPPER])
    cv2.imwrite('out.jpg', img[:,:,[2,1,0]])