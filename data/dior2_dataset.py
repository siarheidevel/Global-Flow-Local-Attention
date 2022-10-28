from numpy.core.fromnumeric import shape
from torch._C import dtype
from torch.functional import Tensor
import torchvision.transforms as transforms
from PIL import Image
import cv2
# import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np
import torch
import torchvision.transforms.functional as F
# import h5py
from util import pose_utils
from tools import prepare_dataset
import math
import numbers
import pandas as pd
from pathlib import Path
import json
from util import util

class SEG:
    labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8}
    BACKGROUND = 0
    HAT = 1
    HAIR = 2    
    FACE = 3
    UPPER = 4
    PANTS = 5
    ARMS = 6
    LEGS = 7
    SHOES = 8


class Dior2Dataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--refit', action=util.StoreList, default=[256 , 192])
        parser.add_argument('--angle', action=util.StoreFloatList, default=None)
        parser.add_argument('--shift', action=util.StoreFloatList, default=None)
        parser.add_argument('--scale', action=util.StoreFloatList, default=None)
        parser.add_argument('--flip', type=int, default=0)
        parser.add_argument('--inpaint_percentage', type=float, default=0.5)
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = Path(opt.dataroot)
        # phase: train or test
        phase = opt.phase
        # pairLstCsv = self.root/f'pairs_{phase}.csv'
        # bonesLstCsv = self.root/f'pose_annotation_filtered_resized_{phase}.csv'

        annotation_index_csv = self.root / 'annotation_index.csv'
        annotation_pairs_csv = self.root / 'annotation_pairs.csv'

        #image size= 192 x 256
        # group;pair1;pair2
        self.pairs_df = pd.read_csv(annotation_pairs_csv, sep=';')
        # 'image_file', 'image_group', 'keypoints_y', 'keypoints_x', 'img_height', 'img_width', 'gender', 'category'
        self.index_df = pd.read_csv(annotation_index_csv, sep=';')
        self.index_df=self.index_df.set_index('image_file')
        self.index_df['image_file']=self.index_df.index

        self.dataset_size = self.pairs_df.shape[0] + int(self.index_df.shape[0] * opt.inpaint_percentage)
        if opt.inpaint_percentage>=1:
            self.dataset_size = self.index_df.shape[0]

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))

        self.trans = transforms.Compose(transform_list)   

    
    def __getitem__(self, index):         
        if index < self.pairs_df.shape[0] and self.opt.inpaint_percentage<1:
            # change pose
            P1_name, P2_name = self.pairs_df.iloc[index][['from','to']].to_list()
            
            # RGB format
            P1_img = cv2.imread(P1_name)[:,:,[2,1,0]]
            BP_1_y,BP_1_x = self.index_df.loc[P1_name][['keypoints_y', 'keypoints_x']].to_list()
            BP_1_array = pose_utils.load_pose_cords_from_strings(BP_1_y,BP_1_x)
            SP1_seg = np.load( P1_name + '.seg.npz')['mask']
            mask = np.ones((self.opt.refit[0],self.opt.refit[1]))
            # RGB format
            P2_img = cv2.imread(P2_name)[:,:,[2,1,0]]
            BP_2_y,BP_2_x = self.index_df.loc[P2_name][['keypoints_y', 'keypoints_x']].to_list()
            BP_2_array = pose_utils.load_pose_cords_from_strings(BP_2_y,BP_2_x)
            SP2_seg = np.load(P2_name + '.seg.npz')['mask']        
            

            # different affine for target and source     
            affine_transform_1 = self._get_affine_stransform(P1_img.shape[0],P1_img.shape[1])
            affine_transform_2 = self._get_affine_stransform(P2_img.shape[0],P2_img.shape[1])
        else:
            #make inpainting
            if self.opt.inpaint_percentage>=1:
                # more stable distribution of images
                rand_self_index = index
            else:
                rand_self_index = np.random.randint(0, self.index_df.shape[0])
            P1_name, BP_1_y, BP_1_x  = self.index_df.iloc[rand_self_index][['image_file','keypoints_y', 'keypoints_x']].to_list()
            P2_name = P1_name
            P1_img = cv2.imread(P1_name)[:,:,[2,1,0]]
            P2_img = P1_img
            BP_1_array = pose_utils.load_pose_cords_from_strings(BP_1_y,BP_1_x)
            BP_2_array = BP_1_array
            SP1_seg = np.load(P1_name + '.seg.npz')['mask']            
            # cv2.imwrite('mask.png',DiorDataset.random_ff_mask(*P1_img.shape[:2], MAXVERTEX = 5, MAX_ANGLE = 60)*255)
            mask = Dior2Dataset.random_ff_mask(*self.opt.refit,MAXVERTEX=8,MAX_ANGLE=60)
            SP2_seg = SP1_seg

            affine_transform_1 = self._get_affine_stransform(P1_img.shape[0],P1_img.shape[1])
            affine_transform_2 = affine_transform_1 
        
        dstSize = (self.opt.refit[1],self.opt.refit[0])

        resized_img1 = cv2.warpAffine(P1_img, affine_transform_1[:2],dstSize, 
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        if P2_name is P1_name and affine_transform_2 is affine_transform_1:
            resized_img2 = resized_img1
        else:
            resized_img2 = cv2.warpAffine(P2_img, affine_transform_2[:2], dstSize,
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        SP1_onehot = Dior2Dataset.obtain_seg_one_hot(SP1_seg, len(SEG.labels))
        if SP2_seg is SP1_seg:
            SP2_onehot = SP1_onehot
        else:
            SP2_onehot = Dior2Dataset.obtain_seg_one_hot(SP2_seg, len(SEG.labels))

        SP1 = cv2.warpAffine(SP1_onehot.transpose(1,2,0), affine_transform_1[:2], dstSize,
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        # clean bckground
        SP1[...,0] = cv2.warpAffine(SP1_onehot.transpose(1,2,0)[...,0], affine_transform_1[:2], dstSize,
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT, borderValue=True)
        #remove white noise
        morphing_kernel = np.ones((3,3), np.uint8)
        SP1 = cv2.dilate(cv2.erode(SP1, morphing_kernel , iterations=1), morphing_kernel , iterations=1)
        SP1 = SP1.transpose(2,0,1)

        if SP2_seg is SP1_seg and affine_transform_2 is affine_transform_1:
           SP2 = SP1
        else:
            SP2 = cv2.warpAffine(SP2_onehot.transpose(1,2,0), affine_transform_2[:2], dstSize,
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            SP2[...,0] = cv2.warpAffine(SP2_onehot.transpose(1,2,0)[...,0], affine_transform_2[:2], dstSize,
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,borderValue=True)
            SP2 = cv2.dilate(cv2.erode(SP2, morphing_kernel , iterations=1), morphing_kernel , iterations=1)
            SP2 = SP2.transpose(2,0,1)  
        

        P1 = self.trans(resized_img1)
        P2 = self.trans(resized_img2)
        BP1 = Dior2Dataset.obtain_bone(BP_1_array, self.opt.refit[0],self.opt.refit[1], affine_transform_1, sigma=6)

        BP1_RGB = self.trans(
            Dior2Dataset.obtain_bone_rgb(BP_1_array,self.opt.refit[0],self.opt.refit[1],affine_transform_1,radius=3))
        if  BP_2_array is BP_1_array  and  affine_transform_2 is affine_transform_1:
            BP2 = BP1
            BP2_RGB =BP1_RGB
        else:
            BP2 = Dior2Dataset.obtain_bone(BP_2_array, self.opt.refit[0],self.opt.refit[1], affine_transform_2, sigma=6)
            BP2_RGB = self.trans(
                Dior2Dataset.obtain_bone_rgb(BP_2_array, self.opt.refit[0],self.opt.refit[1], affine_transform_2,radius=3))
        
        # cv2.imwrite('augm_bones1.jpg',BP1_RGB[...,[2,1,0]])
        # cv2.imwrite('augm_orig1.jpg',cv2.cvtColor(P1_img, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('augm_orig2.jpg',cv2.cvtColor(P2_img, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('augm_img1.jpg',cv2.cvtColor(resized_img1, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('augm_img2.jpg',cv2.cvtColor(resized_img2, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('augm_seg.jpg',SP1[...,0]*255)
        # cv2.imwrite('augm_seg.jpg',SP1[0]*255)
        # # cv2.imwrite('augm_seg.jpg',np.concatenate((SP1_onehot[0],SP1[0]),-1)*254)
        # cv2.imwrite('augm_img1.jpg',cv2.cvtColor((1-SP1[0][...,np.newaxis]) * resized_img1, cv2.COLOR_RGB2BGR) )
        # cv2.imwrite('augm_bones1.jpg',np.sum(BP1, axis=0)*255)
        # cv2.imwrite('augm_bones2.jpg',np.sum(BP2, axis=0)*255)
        # cv2.imwrite('augm_mask.jpg',mask*255)
           
        # print(torch.Tensor(mask).type(torch.uint8).shape)
        return {'P1': P1, 'BP1': torch.Tensor(BP1), 'SP1': torch.Tensor(SP1).type(torch.bool), 
                'P2': P2, 'BP2': torch.Tensor(BP2), 'SP2': torch.Tensor(SP2).type(torch.bool),
                'BP1_RGB': torch.Tensor(BP1_RGB), 'BP2_RGB': torch.Tensor(BP2_RGB),
                'M1' : torch.Tensor(mask).type(torch.bool), 'P1_path': P1_name, 'P2_path': P2_name}


    def __len__(self):
        return self.dataset_size


    def name(self):
        return "dior2"
    

    def _get_affine_stransform(self, img_height, img_width):

        fit_height, fit_width = self.opt.refit[0], self.opt.refit[1]

        center = img_height * 0.5 + 0.5, img_width * 0.5 + 0.5
        do_flip, angle, shift, scale = self.getRandomAffineParam()
        affine_matrix = Dior2Dataset.get_affine_matrix(center=center, fit=(fit_height, fit_width), angle=angle, translate=shift,
                scale=scale, flip=do_flip)
        return affine_matrix


    def getRandomAffineParam(self):
        if self.opt.angle is not None:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not None:
            scale   = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale=1
        if self.opt.shift is not None:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x=0
            shift_y=0
        if self.opt.flip:
            do_flip = np.random.choice([False, True])
        else:
            do_flip = False
        return do_flip, angle, (shift_x,shift_y), scale


    @staticmethod
    def obtain_bone(array, height:int, width:int,  affine_matrix, sigma=6):
        pose  = pose_utils.cords_to_map(array, (height, width), affine_matrix=affine_matrix, sigma=sigma)
        pose = np.transpose(pose,(2, 0, 1))
        return pose 


    @staticmethod
    def obtain_bone_rgb(array, height:int, width:int,  affine_matrix, radius=3):
        af_c=pose_utils.apply_affine_to_coords(array, (height, width), affine_matrix)
        rgb_bones = pose_utils.draw_pose_from_cords(af_c,(height, width),radius=radius)[0]
        return rgb_bones


    @staticmethod
    def obtain_seg_one_hot(segmentation, class_count:int):
        one_hot = np.zeros((segmentation.size, class_count), dtype=np.uint8)
        one_hot[np.arange(segmentation.size),segmentation.ravel()] = 1
        one_hot.shape = segmentation.shape + (class_count,)
        one_hot = np.transpose(one_hot,(2, 0, 1))
        return one_hot


    
    @staticmethod
    def _load_pose_cords_from_strings(y_str, x_str):
        y_cords = json.loads(y_str)
        x_cords = json.loads(x_str)
        return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

    
    @staticmethod
    def get_affine_matrix(center, fit, angle, translate, scale,  flip):
        if flip:
            # https://stackoverflow.com/questions/57863376/why-i-cannot-flip-an-image-with-opencv-affine-transform
            M_x_flip=np.float32([[-1, 0, 2*center[1]-1], [0, 1, 0]])
        else:
            M_x_flip=np.float32([[1, 0, 0], [0, 1, 0]])

        fit_scale = min(fit[0]/(2*center[0]), fit[1]/(2*center[1]))
        FIT_scale = np.float32([[fit_scale, 0, 0], [0, fit_scale, 0]])
        
        M_scale = np.float32([[scale, 0, 0], [0, scale, 0]])

        M_translate = np.float32([[1, 0, translate[0]*fit[0]], [0, 1, translate[1]*fit[1]]])

        rads = math.radians(angle)
        cos, sin = math.cos(rads), math.sin(rads)
        M_rotate = np.float32([[cos, -sin, fit[0]*(1-cos) - fit[1]*sin], 
            [sin, cos, fit[1]*(1-cos) + fit[0]*sin]])
        M_rotate = cv2.getRotationMatrix2D((fit[0]/2,fit[1]/2), angle, 1.0)
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

        M_x_flip = np.vstack((M_x_flip,[0,0,1]))
        M_rotate = np.vstack((M_rotate,[0,0,1]))
        M_translate = np.vstack((M_translate,[0,0,1]))
        M_scale = np.vstack((M_scale,[0,0,1]))
        FIT_scale = np.vstack((FIT_scale,[0,0,1]))

        return M_translate @ M_scale  @ FIT_scale @ M_rotate @ M_x_flip
        # return M_x_flip @ M_rotate @ M_scale @ M_translate


    @staticmethod
    def random_ff_mask(image_height:int ,image_width:int, MAXVERTEX:int = 5, MAX_ANGLE:float = 60,
                       MAX_LENGTH = 0, MAX_BRUSH_WIDTH = 0):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        if MAX_LENGTH == 0:
            MAX_LENGTH =0.7 * min(image_height,image_width)
        
        if MAX_BRUSH_WIDTH == 0:
            MAX_BRUSH_WIDTH = 0.03 * min(image_height,image_width)

        config = {
            'img_shape':(image_height, image_width),
            'mv': MAXVERTEX,
            'ma': MAX_ANGLE,
            'ml': MAX_LENGTH,
            'mbw': MAX_BRUSH_WIDTH
        }

        h,w = config['img_shape']
        mask = np.zeros((h,w))
        num_v = 5 + np.random.randint(config['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)


        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(config['ma'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(config['ml'])
                brush_w = 10+np.random.randint(config['mbw'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return 1-mask.reshape(mask.shape).astype(np.float32)


