import os
import shutil
from PIL import Image
import re

IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
	root_fashion_images = '/home/deeplab/datasets/deepfashion/inshop/small/img'

	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	# new_root = './fashion'
	# if not os.path.exists(new_root):
	# 	os.mkdir(new_root)

	train_root = os.path.join(dir, 'train')
	if not os.path.exists(train_root):
		os.mkdir(train_root)

	test_root = os.path.join(dir, 'test')
	if not os.path.exists(test_root):
		os.mkdir(test_root)

	train_images = []
	train_f = open(os.path.join(dir, 'train.lst'), 'r')
	for lines in train_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			train_images.append(lines)

	test_images = []
	test_f = open(os.path.join(dir, 'test.lst'), 'r')
	for lines in test_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			test_images.append(lines)

	# print(train_images, test_images)

# 	for root, _, fnames in sorted(os.walk(os.path.join(dir, 'img_highres'))):
# 		for fname in fnames:
# 			if is_image_file(fname):
# 				path = os.path.join(root, fname)
# 				path_names = path.split('/') 
# 				print(path_names)

# 				path_names = path_names[6:]# path_names = path_names[2:]
# 				del path_names[1]
# 				path_names[3] = path_names[3].replace('_', '')
# 				path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
# 				path_names = "".join(path_names)
# 				# img = Image.open(path)
# 				if path_names in train_images:
# 					shutil.copy(path, os.path.join(train_root, path_names))
# 					print(os.path.join(train_root, path_names))
# # 					pass
# 				elif path_names in test_images:
# 					shutil.copy(path, os.path.join(test_root, path_names))
# 					print(os.path.join(train_root, path_names))
# # 					pass
	print(f' train size={len(train_images)}, test size={len(test_images)}')
	for item in train_images:
		# from_ = os.path.join(root_fashion_dir, item)
		from_ = root_fashion_images + re.sub(r'(fashion)(WOMEN|MEN)(.+)(id)(\d{8})(\d{2}_\d)',r'/\2/\3/\4_\5/\6_',item)
		if not os.path.isfile(from_):
			print('not found:', from_)
		to_ = os.path.join(train_root, item)
		os.system('cp %s %s' %(from_, to_))
	
	for item in test_images:
		# from_ = os.path.join(root_fashion_dir, item)
		from_ = root_fashion_images + re.sub(r'(fashion)(WOMEN|MEN)(.+)(id)(\d{8})(\d{2}_\d)',r'/\2/\3/\4_\5/\6_',item)
		if not os.path.isfile(from_):
			print('not found:', from_)
		to_ = os.path.join(test_root, item)
		os.system('cp %s %s' %(from_, to_))

# make_dataset('./dataset/fashion/')
make_dataset("/home/deeplab/datasets/deepfashion/inshop/small")
