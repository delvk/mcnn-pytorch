import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import cv2
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = True
save_output = False

image_path = 'othertest/test.1.jpg'
model_path = 'saved_models/mcnn_SH_B_95.h5'

# output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]

# file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
# net.cuda()
# net.eval()
mae = 0.0
mse = 0.0

#load test data
# data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)
img = cv2.imread(image_path, 0)

img = img.astype(np.float32, copy=False)
ht = img.shape[0]
wd = img.shape[1]
ht_1 = int((ht/4)*4)
wd_1 = int((wd/4)*4)
img = cv2.resize(img, (int(wd_1), int(ht_1)))


img = img.reshape((1, 1, img.shape[0], img.shape[1]))

density_map = net(img)
density_map = density_map.data.cpu().numpy()
et_count = np.sum(density_map)
print("ET: {}".format(et_count))
utils.display_results_alt(img, density_map)