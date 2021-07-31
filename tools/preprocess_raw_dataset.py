"""
Prepare openpose 18 points in filen_name.pose.txt
Prepare clothes parsing in filename.seg.npz
Prepare segmented image + pose points in filename.parsed.png
"""
import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageColor, ImageDraw
from pathlib import Path
import glob
import logging
# def add_path(path):
#     import sys
#     if path not in sys.path:
#         sys.path.insert(0, path)


# add_path('/home/deeplab/devel/Global-Flow-Local-Attention')

import generate_mask_features as mask_features
# import generate_pose_features as pose_features
# https://paperswithcode.com/sota/keypoint-detection-on-coco

def _init():
    pass

def process_file_image(image_file: Path):
    """
    Prepare openpose 18 points in filen_name.pose.txt
    Prepare clothes parsing in filename.seg.npz
    Prepare segmented image + pose points in filename.render.jpg
    """
    seg_file_name = image_file.parent / (str(image_file.name) + '.seg')
    # pose_file = image_file.parent / (str(image_file.name) + '.pose.txt')
    if Path(str(seg_file_name)+'.npz').exists():# and pose_file.exists():
        return

    # {'mask_rgb_image': mask_rgb, 'mask_array': reduced_mask, 'rendered': rendered}
    mask_res = mask_features.parse_human_file(str(image_file))
    
    
    #save seg mask array
    seg_file_name = image_file.parent / (str(image_file.name) + '.seg')
    np.savez_compressed(str(seg_file_name), mask = mask_res['mask_array'].astype('uint8'))

    # save pose points
    with open(pose_file, 'w') as f:
        text_coords=f"{str(cords[:,0].tolist())}\t{str(cords[:,1].tolist())}\t{image_height}\t{image_width}"
        f.write(text_coords)
    
    # save rendered paring output
    render = mask_res['rendered']#np.concatenate((rendered[:, :, [2, 1, 0]],mask_res['rendered'] ), axis=1)
    render = render[:,:render.shape[1]//3]
    render_file = image_file.parent / (str(image_file.name) + '.render.jpg')
    cv2.imwrite(str(render_file), cv2.resize(render.astype('float32'),None, fx=0.5, fy=0.5,
        interpolation=cv2.INTER_AREA) ,[int(cv2.IMWRITE_JPEG_QUALITY), 50])


def process_file_image_only_seg(image_file: Path):
    """
    Prepare openpose 18 points in filen_name.pose.txt
    Prepare clothes parsing in filename.seg.npz
    Prepare segmented image + pose points in filename.render.jpg
    """
    seg_file_name = image_file.parent / (str(image_file.name) + '.seg')
    if Path(str(seg_file_name)+'.npz').exists():
        return

    # {'mask_rgb_image': mask_rgb, 'mask_array': reduced_mask, 'rendered': rendered}
    mask_res = mask_features.parse_human_file(str(image_file))
    
    
    #save seg mask array
    seg_file_name = image_file.parent / (str(image_file.name) + '.seg')
    np.savez_compressed(str(seg_file_name), mask = mask_res['mask_array'].astype('uint8'))

    # save rendered paring output
    render = mask_res['rendered']#np.concatenate((rendered[:, :, [2, 1, 0]],mask_res['rendered'] ), axis=1)
    # render[:,:rendered.shape[1]]=rendered[:, :, [2, 1, 0]]
    render_file = image_file.parent / (str(image_file.name) + '.render.jpg')
    cv2.imwrite(str(render_file), cv2.resize(render.astype('float32'),None, fx=0.5, fy=0.5,
        interpolation=cv2.INTER_AREA) ,[int(cv2.IMWRITE_JPEG_QUALITY), 50])

    
def run_fast_scandir(dir, ext, exclude_pattern:str = None):    # dir: str, ext: list
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                if exclude_pattern is None or not exclude_pattern in f.name:
                    files.append(f.path)
    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext, exclude_pattern)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


# process_file_image(Path("/home/deeplab/datasets/custom_fashion/data/wildberries_ru/7671611/7671611-1.jpg"))

def process_image_dir(image_dir: Path, from_seq=0, to_seq=0):
    image_file_list = glob.glob(str(image_dir) + '/**/*.jpg', recursive=True)
    # image_file_list = run_fast_scandir(str(image_dir),[".jpg"],exclude_pattern='.render.jpg' )[1]
    for index, image_file in enumerate(image_file_list):
        if image_file.endswith(".render.jpg"):
            continue
        if index<from_seq: continue
        if to_seq>0 and index>=to_seq: break
        if index % 1000 == 0:
            print(f"{index}/{len(image_file_list)}  {image_file}")
        try:
            process_file_image_only_seg(Path(image_file)) 
        except Exception as err:
            logging.error(f'Exception {image_file}: {err}')

if __name__ == '__main__':
    # process_image_dir(Path("/home/deeplab/datasets/custom_fashion/data/wildberries_ru"))
    # process_image_dir(Path("/home/deeplab/datasets/custom_fashion/data"))
    process_image_dir(Path('/home/deeplab/datasets/deepfashion/diordataset_custom/img_highres'))
# process_image_dir(Path("/home/deeplab/datasets/custom_fashion/data/lamoda_ru"), from_seq=200000, to_seq=300000)
# process_image_dir(Path("/home/deeplab/datasets/custom_fashion/data/lamoda_ru"))




# /home/deeplab/datasets/custom_fashion/data/wildberries_ru/29552185/29552185-1.jpg
