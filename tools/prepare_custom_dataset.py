from pathlib import Path
import logging
import cv2
import numpy as np
from typing import List
import pandas as pd
import json
import re
import glob
from itertools import permutations
import numpy as np

logging.basicConfig(level = logging.INFO)

def _check_valid_pose(points: List, must_be_points: List = [['Rhip', 'Lhip', 'Lsho', 'Rsho']],
                      is_only_frontal: bool = False) -> bool:
    LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
              'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Reye', 'Leye', 'Rear',  'Lear']
    MISSING_VALUE = -1
    point_names = {}
    for i, name in enumerate(LABELS):
        if points[i][0] != MISSING_VALUE and points[i][1] != MISSING_VALUE:
            point_names[name] = points[i][::-1]
    valid_pose = False
    if must_be_points is not None and len(must_be_points) > 0:
        for variant in must_be_points:
            has_variant = True
            for name in variant:
                has_variant = has_variant and (name in point_names)
            valid_pose = valid_pose or has_variant

    if valid_pose and is_only_frontal:
        if 'Lsho' in point_names \
                and 'Rsho' in point_names:
            if point_names['Lsho'][0] > point_names['Rsho'][0]:
                # check if left shoulder coordinate > right shoulder coordinate
                valid_pose = valid_pose and True
            else:
                valid_pose = valid_pose and False
        elif 'Lhip' in point_names \
                and 'Rhip' in point_names:
            if point_names['Lhip'][0] > point_names['Rhip'][0]:
                # check if left hip coordinate > right hip coordinate
                valid_pose = valid_pose and True
            else:
                valid_pose = valid_pose and False
        else:
            valid_pose = False

    return valid_pose


def _load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def prepare_pose_annotation(images_dir: Path):
    """
    image_file, image_group, y_pose_coords, x_pose_coords, img_height, img_width, gender, category
    """
    pose_annotation_list = []

    """
    image_group:[image_files]
    """
    pose_groups = {}

    pose_file_list = glob.glob(str(images_dir) + '/**/*.pose2.txt', recursive=True)
    for index, pose_file in enumerate(pose_file_list):
        try:
            if pose_file == '/home/deeplab/datasets/custom_fashion/data/demo/15447638/15447638-5.jpg.pose2.txt':
                print(pose_file)
            image_file_name = re.search(r'.+/(.+?)\.pose2\.txt', pose_file).group(1)
            image_group_id = Path(pose_file).parent.name
            if '/deepfashion/' in str(images_dir):
                #fix for deep fashion group
                image_group_id +='_' + re.search(r'(\d+)_.+', image_file_name).group(1)

            image_file = Path(pose_file).parent / image_file_name
            if not image_file.exists():
                logging.warning(f' No image {image_file_name} found for pose file {pose_file}')
                continue

            # seg_file = Path(pose_file).parent / (image_file_name + '.seg3.render.png')
            seg_file = Path(pose_file).parent / (image_file_name + '.seg_qanet.render.png')
            if not seg_file.exists():
                logging.warning(f' No segmentation {seg_file} found for pose file {pose_file}')
                continue

            # seg = np.load(seg_file)['mask']
            # if np.max(seg)>8:
            #     logging.warning(f' Wrong segmentation {seg_file} found for pose file {pose_file}')
            #     continue

            
            with open(pose_file) as f:
                points_y,points_x,height,width = f.readline().split('\t')
            points = _load_pose_cords_from_strings(points_y, points_x)

            is_valid_pose = _check_valid_pose(points,
                must_be_points=[['Rhip', 'Lhip', 'Lsho', 'Rsho'], ['Rhip', 'Lhip', 'Rkne', 'Lkne']],
                is_only_frontal=True)

            if not is_valid_pose:
                # add back full figure
                is_valid_pose = _check_valid_pose(points,
                    must_be_points=[['Rhip', 'Lhip', 'Lsho', 'Rsho', 'Rkne', 'Lkne', 'nose']],
                    is_only_frontal=False)

            
            if not is_valid_pose:
                # logging.info(f' No valid pose for pose file {pose_file}')
                continue

            gender, category = '', ''
            try:
                with open(Path(pose_file).parent / 'meta.json') as f:
                    meta = json.load(f)
                    category = meta.get('name','')
                    gender = meta.get('gender','')
            except:
                try:
                    gender = {'MEN':'man', 'WOMEN':'woman'}[image_file.parent.parent.parent.name]
                    category = image_file.parent.parent.name
                except:
                    logging.warning(f' No meta for pose file {pose_file}')

            pose_annotation_list.append([image_file, image_group_id, points_y,points_x, height,width,gender, category])
            pose_groups[image_group_id] = pose_groups.get(image_group_id, []) + [image_file]
            if index%1000==0:
                logging.info(f'{len(pose_annotation_list)} from {index} / {len(pose_file_list)}')

        except Exception as err:
            logging.exception(f'{pose_file}, {err}')
    
    logging.info(f'Total valid files {len(pose_annotation_list)} from {len(pose_file_list)}')
    logging.info(f'Total valid groups {len(pose_groups)}')

    # saving annotation csv
    anno_df=pd.DataFrame(data=pose_annotation_list,
        columns=['image_file', 'image_group', 'keypoints_y', 'keypoints_x', 'img_height', 'img_width', 'gender', 'category'])
    
    
    # creating permutation pairs/ limit 5 per group
    LIMIT_PER_GROUP_PAIRS = 8
    LIMIT_PAIRS = 10000000
    pairs_list =[]
    for group, image_files in pose_groups.items():
        if len(pairs_list) > LIMIT_PAIRS:
            break
        if len(image_files)==0:
            continue
        pairs = list(permutations(sorted(image_files), 2))[:LIMIT_PER_GROUP_PAIRS]
        for p in pairs:
            pairs_list.append([group, p[0], p[1]])
    
    pairs_anno_df = pd.DataFrame(data=pairs_list,
        columns=['group', 'from', 'to'])
    logging.info(f'Total pairs {len(pairs_anno_df)}')
    anno_df.to_csv(images_dir/ 'annotation_index_qanet.csv', index=False, sep=';')
    pairs_anno_df.to_csv(images_dir/ 'annotation_pairs_qanet.csv', index=False, sep=';')    
        

def stat_dataset(anno_file:Path):
    anno_df = pd.read_csv(anno_file, sep=';')
    print(anno_df['gender'].value_counts())
    print(anno_df['category'].value_counts())
    print(anno_df.reset_index()[['index','gender','category']].groupby(['category'])
        .count().sort_values(by='gender',ascending=False).reset_index()[:100].values)
    print(anno_df.reset_index()[['index','gender','category']]
        .groupby(['gender','category']).count().reset_index()
        .sort_values(by=['index'],ascending=False)[:100].values)
    print(anno_df.info())

if __name__ == "__main__":
    # TODO /home/deeplab/datasets/custom_fashion/demo
    # prepare_pose_annotation(Path('/home/deeplab/datasets/custom_fashion/demo'))
    # prepare_pose_annotation(Path('/home/deeplab/datasets/custom_fashion/data/'))
    # stat_dataset(Path('/home/deeplab/datasets/custom_fashion/data/annotation_index.csv'))
    prepare_pose_annotation(Path('/home/deeplab/datasets/deepfashion/diordataset_custom'))
    stat_dataset(Path('/home/deeplab/datasets/deepfashion/diordataset_custom/annotation_index_qanet.csv'))

    
    print('finished')
    