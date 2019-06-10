import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils

# print('hello')
# print(torch.cuda.get_device_name(0))
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path =  '/home/jake/Desktop/Projects/Python/dataset/SH_B/cooked/test_10/images'
gt_path = '/home/jake/Desktop/Projects/Python/dataset/SH_B/cooked/test_10/ground_truth'
model_path = 'saved_models/mcnn_SH_B_110.h5'

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
# net.cuda()
# net.eval()
mae = 0.0
mse = 0.0

#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

for blob in data_loader:                        
    im_data = blob['data']
    gt_data = blob['gt_density']
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    print("Ground truth: {:0.2f}, Estimate: {:0.2f}".format(gt_count, et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_results(im_data, gt_data, density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

# f = open(file_results, 'w') 
# f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
# f.close()