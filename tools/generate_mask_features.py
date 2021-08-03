import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageColor, ImageDraw
from pathlib import Path
import glob


def add_path(path):
    import sys
    if path not in sys.path:
        sys.path.insert(0, path)


add_path('/home/deeplab/devel/Self-Correction-Human-Parsing')

lip_dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

    }}

# reduced_dataset = {
#     'labels': {'background': 0, 'hair': 1, 'face': 2, 'upper-clothes': 3, 'pants': 4, 'skirt': 5, 'arm': 6, 'leg': 7},

#     'lip2reduced': {'Background': 'background', 'Hat': 'hair', 'Hair': 'hair', 'Glove': 'arm', 'Sunglasses': 'face', 'Upper-clothes': 'upper-clothes',
#                     'Dress': 'upper-clothes', 'Coat': 'upper-clothes',
#                     'Socks': 'pants', 'Pants': 'pants', 'Jumpsuits': 'pants', 'Scarf': 'upper-clothes', 'Skirt': 'skirt', 
#                     'Face': 'face', 'Left-arm': 'arm', 'Right-arm': 'arm',
#                     'Left-leg': 'leg', 'Right-leg': 'leg', 'Left-shoe': 'leg', 'Right-shoe': 'leg'}
# }

# reduced_dataset = {
#     'labels': {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
#                'pants': 5, 'skirt': 6, 'dress': 7, 'arm': 8, 'leg': 9, 'shoes':10},

#     'lip2reduced': {'Background': 'background', 'Hat': 'hat', 'Hair': 'hair', 'Glove': 'arm', 'Sunglasses': 'face', 'Upper-clothes': 'upper-clothes',
#                     'Dress': 'dress', 'Coat': 'dress',
#                     'Socks': 'shoes', 'Pants': 'pants', 'Jumpsuits': 'dress', 'Scarf': 'upper-clothes', 'Skirt': 'skirt', 
#                     'Face': 'face', 'Left-arm': 'arm', 'Right-arm': 'arm',
#                     'Left-leg': 'leg', 'Right-leg': 'leg', 'Left-shoe': 'shoes', 'Right-shoe': 'shoes'}
# }

# reduced_dataset = {
#     'labels': {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
#                'pants': 5, 'dress': 6, 'arm': 7, 'leg': 8, 'shoes':9},

#     'lip2reduced': {'Background': 'background', 'Hat': 'hat', 'Hair': 'hair', 'Glove': 'arm', 'Sunglasses': 'face', 
#                     'Upper-clothes': 'upper-clothes',
#                     'Dress': 'dress', 'Coat': 'dress',
#                     'Socks': 'shoes', 'Pants': 'pants', 'Jumpsuits': 'dress', 'Scarf': 'upper-clothes', 'Skirt': 'pants', 
#                     'Face': 'face', 'Left-arm': 'arm', 'Right-arm': 'arm',
#                     'Left-leg': 'leg', 'Right-leg': 'leg', 'Left-shoe': 'shoes', 'Right-shoe': 'shoes'}
# }

reduced_dataset = {
    'labels': {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8},

    'lip2reduced': {'Background': 'background', 'Hat': 'hat', 'Hair': 'hair', 'Glove': 'arm', 'Sunglasses': 'face', 
                    'Upper-clothes': 'upper-clothes',
                    'Dress': 'upper-clothes', 'Coat': 'upper-clothes',
                    'Socks': 'shoes', 'Pants': 'pants', 'Jumpsuits': 'upper-clothes', 'Scarf': 'upper-clothes', 'Skirt': 'pants', 
                    'Face': 'face', 'Left-arm': 'arm', 'Right-arm': 'arm',
                    'Left-leg': 'leg', 'Right-leg': 'leg', 'Left-shoe': 'shoes', 'Right-shoe': 'shoes'}
}


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def draw_legend(colors,labels, imsize=(200,600)):
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatches
    # labels = [k for k,v in reduced_dataset['labels'].items()]
    # colors = np.array(get_palette(len(reduced_dataset['labels']))).reshape(len(reduced_dataset['labels']), 3)

    # create a patch (proxy artist) for every color 
    # patches = [ mpatches.Patch(color=colors[i]/255, label=labels[i]) for i in range(len(labels)) ]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # cv2.

    # initialize the legend visualization
    legend = np.zeros((*imsize, 3), dtype="uint8")
    # loop over the class names + colors
    per_row = int(imsize[0]/len(labels))
    for i in range(len(labels)):
        color = colors[i]
        label = labels[i]
        cv2.putText(legend, label, (5, (i * per_row) + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(legend, (100, (i * per_row)), (imsize[1], (i * per_row) + 25),
            list(reversed(color.tolist())), -1)

    return legend


def load_model():
    import networks
    net = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(
        '/home/deeplab/devel/Self-Correction-Human-Parsing/saved_models/exp-schp-201908261155-lip.pth')['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    # if 'cuda'==device and torch.cuda.is_available():
    #     net.cuda()
    net = net.to(device)
    net.eval()
    return net


num_classes = lip_dataset_settings['lip']['num_classes']
rgb_palette_colors = np.array(get_palette(num_classes)).reshape(num_classes, 3)
rgb_reduced_palette_colors = np.array(get_palette(
    len(reduced_dataset['labels']))).reshape(len(reduced_dataset['labels']), 3)
lip2reduced_ids = np.array([(lip_dataset_settings['lip']['label'].index(lip_label),
                            reduced_dataset['labels'][red_label]) for (lip_label, red_label) in reduced_dataset['lip2reduced'].items()]).T
redused_legend = draw_legend(rgb_reduced_palette_colors,[k for k,v in reduced_dataset['labels'].items()],
    (lip_dataset_settings['lip']['input_size'][0],200))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
human_parsing_model = load_model()


# num_classes = self_dataset_settings['lip']['num_classes']
# input_size = self_dataset_settings['lip']['input_size']
# label = self_dataset_settings['lip']['label']


def _xywh2cs(x, y, w, h, aspect_ratio=1):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def parse_human(img):
    from utils.transforms import get_affine_transform
    from utils.transforms import transform_logits

    h, w, _ = img.shape
    person_center, s = _xywh2cs(0, 0, w - 1, h - 1)
    r = 0

    input_size = lip_dataset_settings['lip']['input_size']

    trans = get_affine_transform(person_center, s, r, input_size)
    input = cv2.warpAffine(
        img,
        trans,
        (int(input_size[1]), int(input_size[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                      0.229, 0.224, 0.225])
    ])
    input = transform(input).unsqueeze(0).to(device)
    output = human_parsing_model(input)
    upsample = torch.nn.Upsample(
        size=input_size, mode='bilinear', align_corners=True)
    upsample_output = upsample(
        output[0][-1][0].unsqueeze(0)).squeeze().permute(1, 2, 0)  # CHW -> HWC
    logits_result = transform_logits(upsample_output.data.cpu(
    ).numpy(), person_center, s, w, h, input_size=input_size)
    # print('logits_result', logits_result.shape)
    mask_array = np.argmax(logits_result, axis=2)
    # mask_image = Image.fromarray(np.asarray(mask_array, dtype=np.uint8))
    # mask_image.putpalette(get_palette(num_classes))

    # reduced dataset
    reduced_mask = lip2reduced_ids[1][mask_array]
    mask_rgb = rgb_reduced_palette_colors[reduced_mask]

    rendered = np.clip(img * 0.5 + 0.5 *
                       mask_rgb[:, :, [2, 1, 0]], 0, 254).astype(np.uint8)
    rendered = np.concatenate((img,rendered,mask_rgb[:, :, [2, 1, 0]],cv2.resize(redused_legend, (200,rendered.shape[0]))), axis=1)
    # cv2.imwrite('ou2.png', img * 0.5 + 0.5 *mask_rgb)
    # cv2.imwrite('ou2.png', rendered[:,:,[2,1,0]])
    # cv2.imwrite('ou2.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('ou2.png',rendered)
    return {'mask_rgb_image': mask_rgb, 'mask_array': reduced_mask, 'rendered': rendered}


def parse_human_file(image_file, clean_bg=False):
    if clean_bg:
        img = np.asarray(clean_background(Path(image_file),
                         background_removal_model)['blended'])
    else:
        img = cv2.imread(image_file)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return parse_human(img)


def process_image_dir(images_dir, output_dir, from_seq=0):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_file_list = glob.glob(images_dir + '/**/*.jpg', recursive=True)
    for index, image_file in enumerate(image_file_list):
        if index<from_seq: continue
        res = parse_human_file(image_file)
        image_file_name = Path(image_file).name
        if index % 10 == 0:
            print(f"{index}/{len(image_file_list)}  {image_file}")
        matr_file_name = os.path.join(output_dir, image_file_name)
        rendered_file_name = os.path.join(output_dir, image_file_name)
        # np.save(matr_file_name, res['mask_array'].astype('uint8'))
        np.savez_compressed(matr_file_name, mask = res['mask_array'].astype('uint8'))
        # loaded = np.load(matr_file_name+'.npz')['mask']
        # cv2.imwrite(rendered_file_name, cv2.cvtColor(
        #     rendered, cv2.COLOR_RGB2BGR))
        cv2.imwrite(rendered_file_name, res['rendered'])


if __name__ == "__main__":
    # /home/deeplab/datasets/deepfashion/inshop/adgan/data/semantic_merge3/MEN/Denim/id_00000080/01_1_front.npy
    # parse_human_file(
    #     '/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train/fashionMENDenimid0000008001_1front.jpg')
    # process_image_dir('/home/deeplab/datasets/clothing-co-parsing/photos',
    #                   '/home/deeplab/devel/my_test_images_out')
    # process_image_dir('/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train',
    #                   '/home/deeplab/datasets/deepfashion/inshop/adgan/data/fashion_resize/train_seg')
    process_image_dir('/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train',
                      '/home/deeplab/datasets/deepfashion/inshop/adgan/highres/train_seg2', from_seq=0)
                      

