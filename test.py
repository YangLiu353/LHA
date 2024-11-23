import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os
import cv2
import time
from thop import profile

from utils.LSNet import LSNet

from utils.config import opt

dataset_path = opt.test_path

# Set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# Load the model
model = LSNet()

# Load pre-trained weights
model.load_state_dict(torch.load('/home/ipal/Home/LY/跑跑4/pre1/Net_epoch_best.pth', weights_only=True))
model.cuda()
model.eval()

# params FLOGs
dummy_image = torch.randn(3, 3, opt.testsize, opt.testsize).cuda()
dummy_ti = torch.randn(3, 3, opt.testsize, opt.testsize).cuda()

flops, params = profile(model, inputs=(dummy_image, dummy_ti))

import os

model_path = '/home/ipal/Home/LY/跑跑4/pre1/Net_epoch_best.pth'
model_size = os.path.getsize(model_path)

# Test
test_mae = []
if opt.task == 'RGBT':
    from rgbt_dataset import test_dataset

    test_datasets = ['VT800', 'VT1000', 'VT5000']
elif opt.task == 'RGBD':
    from rgbd_dataset import test_dataset

    test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
else:
    raise ValueError(f"Unknown task type {opt.task}")

for dataset in test_datasets:
    mae_sum = 0
    save_path = '/home/ipal/Home/LY/跑跑4/res1/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    edge_save_path = '/home/ipal/Home/LY/跑跑4/edge/' + dataset + '/'
    if not os.path.exists(edge_save_path):
        os.makedirs(edge_save_path)

    if opt.task == 'RGBT':
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        ti_root = dataset_path + dataset + '/T/'
    elif opt.task == 'RGBD':
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        ti_root = dataset_path + dataset + '/depth/'
    else:
        raise ValueError(f"Unknown task type {opt.task}")
    test_loader = test_dataset(image_root, gt_root, ti_root, opt.testsize)

    total_time = 0
    num_images = 0

    for i in range(test_loader.size):
        image, gt, ti, name, img_size = test_loader.load_data()
        gt = gt.cuda()
        image = image.cuda()
        ti = ti.cuda()
        if opt.task == 'RGBD':
            ti = torch.cat((ti, ti, ti), dim=1)

        # Start timer
        start_time = time.time()

        res , edge1_out = model(image, ti)

        res = F.interpolate(res, img_size, mode='bilinear', align_corners=True)
        gt = F.interpolate(gt, img_size, mode='bilinear', align_corners=True)
        predict = torch.sigmoid(res)
        predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
        mae = torch.sum(torch.abs(predict - gt)) / torch.numel(gt)
        mae_sum += mae.item()

        # End timer
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Accumulate total time
        total_time += elapsed_time
        num_images += 1

        predict = predict.data.cpu().numpy().squeeze()
        cv2.imwrite(save_path + name, predict * 255)

        edge = F.interpolate(edge1_out, img_size, mode='bilinear', align_corners=False)
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        cv2.imwrite(edge_save_path+name, edge*255)



    test_mae.append(mae_sum / test_loader.size)
    avg_fps = num_images / total_time
    print('Dataset:', dataset)
    print('Average FPS:', avg_fps)
    print('MAE:', mae_sum / test_loader.size)
    print('----------------------')

print('Test Done!', 'MAE', test_mae)
print(f"FLOPS: {flops}, Parameters:{params}")
print("model_size：{:.2f} MB".format(model_size / (1024 * 1024)))
