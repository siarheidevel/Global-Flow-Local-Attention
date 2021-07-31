import sys
import cv2
import os
import numpy as np
import pose_utils
import glob
import csv
from pathlib import Path
# from src import util
# from src.model import bodypose_model


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path('/home/deeplab/devel/pytorch-openpose/src')


def load_model():
    from body import Body
    body_estimation = Body(
        '/home/deeplab/devel/pytorch-openpose/model/body_pose_model.pth')
    return body_estimation


body_model = load_model()


def process_image(rgb_image, do_render=True, do_matrix=True,radius =3):
    '''
    returns tuple 
    pose coordinates x[],y[]
    numpy array[H,W,Channels18]
    point on image [H,W,3]
    '''
    cords = -1*np.ones((18, 2)).astype(int)
    coordinates, persons = body_model(rgb_image)
    # print(persons)
    # print(coordinates) result_image_index = np.argmax(subset[:, -2])
    """
    coordinates: [x, y, score, id]
    persons: n*[20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts]    
    """
    if persons.shape[0] > 0:
        # detect only one person
        best_person_id = np.argmax(persons[:, -2])

        part_indexes = np.nonzero(persons[best_person_id, :18] >= 0)[0].astype(int)
        part_array_idx = persons[best_person_id, :18][part_indexes].astype(int)
        c = coordinates[part_array_idx, :2][:, [1, 0]].astype(int)
        cords[part_indexes] = c
    cords_map = None
    if do_matrix:
        cords_map = pose_utils.cords_to_map(cords, rgb_image.shape[:2])
    rendered = None
    if do_render:
        colors, mask = pose_utils.draw_pose_from_cords(
            cords, rgb_image.shape[:2], radius=radius)
        rendered = rgb_image * \
            (colors[..., ] == [0, 0, 0]) + \
            colors * (colors[..., ] != [0, 0, 0])
    return cords, cords_map, rendered


def process_image_file(image_file, **kwargs):
    oriImg = cv2.imread(image_file)
    # B,G,R order
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    return (oriImg.shape[0],oriImg.shape[1],*process_image(oriImg, **kwargs))


def process_image_dir(images_dir, output_dir):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_file_list = glob.glob(images_dir + '/**/*.jpg', recursive=True)
    csv_file_name = os.path.join(output_dir, 'pose_annotation.csv')
    with open(csv_file_name, 'w', newline='') as csvfile:
        coord_writer = csv.writer(csvfile, delimiter=':',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for index, image_file in enumerate(image_file_list):
            image_height, image_width, cords, cords_map, rendered = process_image_file(image_file)
            image_file_name = Path(image_file).name
            coord_writer.writerow([image_file, str(cords[:,0].tolist()), str(cords[:,1].tolist()),image_width, image_height])
            if index % 10 == 0:
                cvs_str = f"{image_file}\t{str(cords[:,0].tolist())}\t{str(cords[:,1].tolist())}"
                print(f"{index}/{len(image_file_list)}  {cvs_str}")
            matr_file_name = os.path.join(output_dir, image_file_name)
            rendered_file_name = os.path.join(output_dir, image_file_name)
            # np.savez_compressed(matr_file_name, pose_map = cords_map.astype('float16'))
            # loaded = np.load(matr_file_name+'.npz')['pose_map']
            cv2.imwrite(rendered_file_name, cv2.cvtColor(
                rendered, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    # process_image_dir('/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train',
    #     '/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train_pose2')

    process_image_dir('/home/deeplab/datasets/custom_fashion/data/wildberries_ru',
        '/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train_pose2')
    
    # process_image_dir('/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train','/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train_pose')
    
    # cords, cords_map, rendered = process_image_file(body_model,
    #                                                 '/home/deeplab/devel/pytorch-openpose/images/demo.jpg')
    # #    '/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train/fashionMENTees_Tanksid0000404701_7additional.jpg')
    # #    '/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train/fashionWOMENCardigansid0000045003_2side.jpg')
    # cv2.imwrite('ou1.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    # print(cords.T)
