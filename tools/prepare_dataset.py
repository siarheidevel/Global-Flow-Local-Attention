from pathlib import Path
import cv2
import numpy as np
from typing import List
import pandas as pd
import json
import re
from itertools import permutations
import math
import numbers
import torchvision.transforms.functional as F
from scipy.ndimage import affine_transform


def _get_affine_matrix(original_size, target_size, preserve_aspect: bool = False):
    """
    returns affine params
    (angle,scale,translate,shear)
    """
    hs, ws = original_size
    wt, ht  = target_size
    source_points = np.float32([[0, 0], [ws/2, hs/2], [ws, 0]])
    if preserve_aspect:
        min_ratio = min(wt/ws, ht/hs)
        wn, hn = ws * min_ratio, hs * min_ratio
        w_offset, h_offset = (wt - wn) / 2, (ht - hn) /2
        target_points = np.float32([[w_offset, h_offset], [wt/2, ht/2], [wt - w_offset, h_offset]])
    else:
        target_points = np.float32([[0, 0], [wt/2, ht/2], [wt, 0]])
    affine_matrix = cv2.getAffineTransform(source_points, target_points)
    return affine_matrix


def _resize_img(img, size=(256, 256), preserve_aspect: bool = False):
    affine_matrx = _get_affine_matrix(img.shape[:2], size, preserve_aspect)
    out_img = cv2.warpAffine(
        img, affine_matrx, dsize=size, interpolation=cv2.INTER_AREA)


# def _resize_img(img, size=(256,256), preserve_aspect:bool = False):
#     h, w = img.shape[:2]
#     c = None if len(img.shape) < 3 else img.shape[2]
#     interpolation = cv2.INTER_AREA if size[0]<h else cv2.INTER_CUBIC
#     if not preserve_aspect:
#         return cv2.resize(img, size, interpolation)
#     min_ratio = min(h/size[0],w/size[1])
#     dsize = (h*min_ratio,w*min_ratio)
#     out_img = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
#     if c is None:
#         mask = np.zeros(size, dtype=img.dtype)
#         mask[(size[0]-dsize[0])/2:(size[0]+dsize[0])/2,
#                 (size[1]-dsize[1])/2:(size[1]+dsize[1])/2] = out_img
#     else:
#         mask = np.zeros((*size, c), dtype=img.dtype)
#         mask[(size[0]-dsize[0])/2:(size[0]+dsize[0])/2,
#                 (size[1]-dsize[1])/2:(size[1]+dsize[1])/2] = out_img
#     return mask





def _check_valid_pose(points: List, must_be_points: List = ['Rhip', 'Lhip', 'Lsho', 'Rsho'],
                      is_only_frontal: bool = False):
    LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
              'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
    MISSING_VALUE = -1
    point_names = {}
    for i, name in enumerate(LABELS):
        if points[i][0] != MISSING_VALUE and points[i][1] != MISSING_VALUE:
            point_names[name] = points[i][::-1]
    valid_pose = True
    if must_be_points is not None and len(must_be_points) > 0:
        for name in must_be_points:
            valid_pose = valid_pose and (name in point_names)
    if valid_pose and is_only_frontal:
        if 'Lsho' in point_names \
                and 'Rsho' in point_names \
                and point_names['Lsho'][0] > point_names['Rsho'][0]:
            # check if left shoulder coordinate > right shoulder coordinate
            valid_pose = valid_pose and True
        elif 'Lhip' in point_names \
                and 'Rhip' in point_names \
                and point_names['Lhip'][0] > point_names['Rhip'][0]:
            # check if left hip coordinate > right hip coordinate
            valid_pose = valid_pose and True
        else:
            valid_pose = False

    return valid_pose


def _load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def filter_pose(pose_csv_file: Path, filtered_csv_file: Path, segments_dir: Path, images_dir: Path):
    df_keypoints = pd.read_csv(pose_csv_file, sep=':', names=[
                               'name', 'keypoints_y', 'keypoints_x', 'width', 'height'])
    df_keypoints['name'] = df_keypoints.apply(
        lambda x: Path(x['name']).name, axis=1)

    def check_valid(x):
        kp_array = _load_pose_cords_from_strings(
            x['keypoints_y'], x['keypoints_x'])
        valid = _check_valid_pose(kp_array, is_only_frontal=True)
        valid = valid and (segments_dir / (x['name'] + '.npz')).exists()
        valid = valid and (images_dir/x['name']).exists()
        return valid
    df_filtered = df_keypoints[df_keypoints.apply(check_valid, axis=1)].copy()
    df_filtered['name'] = df_keypoints['name']
    df_filtered.to_csv(path_or_buf=str(filtered_csv_file),
                       sep=':', index=False, header=None)


def build_image_pairs(pose_csv_file: Path, pairs_csv_file: Path):
    df_keypoints = pd.read_csv(pose_csv_file, sep=':', names=[
                               'name', 'keypoints_y', 'keypoints_x', 'width', 'height'])

    def find_id(name):
        return int(re.findall(r'id(\d{8,})_', name)[0])
    persons = df_keypoints.apply(lambda x: find_id(x['name']), axis=1)
    df_keypoints['person'] = persons
    fr, to = [], []
    for person in pd.unique(persons):
        pairs = list(permutations(
            df_keypoints[df_keypoints['person'] == person]['name'], 2))
        # pairs = zip(*list(permutations(df[df['person'] == person]['name'], 2)))
        if len(pairs) != 0:
            fr += [p[0] for p in pairs]
            to += [p[1] for p in pairs]
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    pair_df.to_csv(path_or_buf=str(pairs_csv_file), index=False, header=None)


def resize(image_file: str, src_image_dir: Path, src_segment_dir: Path,
           target_image_dir:Path, target_segment_dir:Path, pose_coordinates,
           target_size=(256, 256), keep_aspect_ratio: bool = True):
    src_img = cv2.imread(str(src_image_dir/image_file))
    seg_matrix = np.load(str(src_segment_dir/(image_file+'.npz')))['mask']

    affine_matrx = _get_affine_matrix(
        src_img.shape[:2], target_size, keep_aspect_ratio)
    resized_img = cv2.warpAffine(
        src_img, affine_matrx, target_size, flags=cv2.INTER_AREA, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128))
    
    # INTER_NEAREST - for classes
    resized_seg = cv2.warpAffine(
        seg_matrix, affine_matrx, target_size, flags=cv2.INTER_NEAREST, 
        borderMode=cv2.BORDER_CONSTANT)
    
    resized_coordinates =[]
    MISSING_VALUE = -1
    for _, point in enumerate(pose_coordinates):
        # point_ =np.dot(affine_matrx, np.matrix([point[1], point[0], 1]).reshape(3,1))
        if point[1] == MISSING_VALUE:
            resized_coordinates.append([MISSING_VALUE, MISSING_VALUE])
        else:
            point_ = affine_matrx @ np.array([point[1], point[0], 1])
            point_0 = int(point_[1])
            point_1 = int(point_[0])
            resized_coordinates.append([point_0,point_1])
    
    # check: blend 3 data in one image
    if False:
        demo_img = resized_img.copy()
        for p in resized_coordinates:
            if p[0] != MISSING_VALUE:
                demo_img = cv2.circle(demo_img, (p[1],p[0]), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('affine_warp.jpg', demo_img)
        cv2.imwrite('affine_warp_seg.png', resized_seg*20)
        # np.repeat(resized_seg[:,:,np.newaxis], 3, axis=2).shape
        cv2.imwrite('affine_warp_seg_blended.jpg',0.5* demo_img +  resized_seg[:,:,np.newaxis]*10)

    cv2.imwrite(str(target_image_dir/image_file), resized_img)
    np.savez_compressed(str(target_segment_dir/image_file), mask = resized_seg.astype('uint8'))

    resized_seg = np.repeat(resized_seg[:,:,np.newaxis], 3, axis=2)
    resized_seg = resized_seg *20
    for p in resized_coordinates:
            if p[0] != MISSING_VALUE:
                resized_seg = cv2.circle(resized_seg, (p[1],p[0]), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(str(target_segment_dir/image_file)+'.png',  resized_seg)
    return pd.Series([np.array(resized_coordinates)[:,0].tolist(), np.array(resized_coordinates)[:,1].tolist()])

def resize_imgs(pose_csv_file: Path, src_image_dir: Path, src_segment_dir: Path,
                target_image_dir:Path, target_segment_dir:Path,
                new_size=(256, 256), preserve_aspect: bool = False):
    df_keypoints = pd.read_csv(pose_csv_file, sep=':', names=[
                               'name', 'keypoints_y', 'keypoints_x', 'width', 'height'])
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_segment_dir.mkdir(parents=True, exist_ok=True)
    transfored_keypoints = df_keypoints.apply(lambda x: resize(x['name'], 
        src_image_dir, src_segment_dir,
        target_image_dir,target_segment_dir,
        _load_pose_cords_from_strings(x['keypoints_y'], x['keypoints_x']), 
        new_size, preserve_aspect), axis=1)
    df_keypoints['keypoints_x']=transfored_keypoints[0]
    df_keypoints['keypoints_y']=transfored_keypoints[1]
    df_keypoints.to_csv(path_or_buf=str(pose_csv_file.parent/'pose_annotation_filtered_resized_train.csv'),
        index=False, header=None, sep=':')
    print(f'Resized {len(transfored_keypoints)} images from {src_image_dir}, to {target_image_dir}')


if __name__ == "__main__":
    # 1 filter valid images: check presence of segments, pose , image
    # 2 generate pairs
    # 3 resize images, keypoints, segments to desired size
    target_dir = Path('/home/deeplab/datasets/deepfashion/diordataset256_176')
    target_dir.mkdir(parents=True, exist_ok=True)

    pose_csv_file = Path(
        '/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train_pose2/pose_annotation.csv')
    src_images_dir = Path(
        '/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train')
    src_segment_dir = Path(
        '/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train_seg2')
    # filter_pose(Path('/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train_pose2/pose_annotation.csv'),
    #     target_dir/ 'pose_annotation_filtered_train.csv',src_segment_dir ,src_images_dir)
    # build_image_pairs(target_dir/ 'pose_annotation_filtered_train.csv',target_dir/ 'pairs_train.csv')

    resize_imgs(target_dir/ 'pose_annotation_filtered_train.csv', src_images_dir, src_segment_dir,
                target_dir/'train'/'img', target_dir/'train'/'seg',
                new_size=(176, 256), preserve_aspect=True)