import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch

class FashionDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 176))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser



    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'fasion-pairs-%s.csv' % phase)
        name_pairs = self.init_categories(pairLst)
        
        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' % phase)
        return image_dir, bonesLst, name_pairs


    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        # for i in range(size):
        #     pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
        #     pairs.append(pair)
        pairs = [tuple(x) for x in pairs_file_train[['from','to']].to_numpy()]

        print('Loading data pairs finished ...')  
        return pairs    

    def name(self):
        return "FashionDataset"

                