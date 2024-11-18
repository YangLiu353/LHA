import os
import torch
import torch.nn.functional as F

import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from utils.config import opt
from torch.cuda import amp



# set the device for training
cudnn.benchmark = True
cudnn.enabled = True


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model
from utils.LSNet import LSNet
model = LSNet()
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
from utils.edge_label import label_edge_prediction
# set the path
train_dataset_path = opt.train_root

val_dataset_path = opt.val_root

save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
if opt.task == 'RGBD':
    from utils.rgbd_dataset import get_loader, test_dataset
    image_root = train_dataset_path + '/rgb/'
    ti_root = train_dataset_path + '/depth/'
    gt_root = train_dataset_path + '/gt/'
    val_image_root = val_dataset_path + '/rgb/'
    val_ti_root = val_dataset_path + '/depth/'
    val_gt_root = val_dataset_path + '/gt/'
else:
    raise ValueError(f"Unknown task type {opt.task}")

train_loader = get_loader(image_root, gt_root, ti_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root,val_ti_root, opt.trainsize)
total_step = len(train_loader)
# print(total_step)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Model:")
logging.info(model)

logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
import torch.nn as nn

class IOUBCE_loss(nn.Module):
    def __init__(self):
        super(IOUBCE_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs,targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b


CE = torch.nn.BCEWithLogitsLoss().cuda()
IOUBCE = IOUBCE_loss().cuda()
class IOUBCEWithoutLogits_loss(nn.Module):
    def __init__(self):
        super(IOUBCEWithoutLogits_loss, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, target_scale):
        b,c,h,w = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, target_scale):

            bce = self.nll_lose(inputs,targets)

            inter = (inputs * targets).sum(dim=(1, 2))
            union = (inputs + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b
IOUBCEWithoutLogits = IOUBCEWithoutLogits_loss().cuda()


step = 0
writer = SummaryWriter(save_path + 'summary', flush_secs = 30)
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()



# BBA
def tesnor_bound(img, ksize):

    '''
    :param img: tensor, B*C*H*W
    :param ksize: tensor, ksize * ksize
    :param 2patches: tensor, B * C * H * W * ksize * ksize
    :return: tensor, (inflation - corrosion), B * C * H * W
    '''

    B, C, H, W = img.shape
    pad = int((ksize - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant',value = 0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion



# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, tis) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            tis = tis.cuda()
            gts = gts.cuda()
            edges = label_edge_prediction(gts)
            if opt.task == 'RGBD':
                tis = torch.cat((tis, tis, tis), dim=1)

            gts2 = F.interpolate(gts, (256, 256))
            gts3 = F.interpolate(gts, (128, 128))
            gts4 = F.interpolate(gts, (64, 64))
            gts5 = F.interpolate(gts, (32, 32))


            edges2 = F.interpolate(edges, (256, 256))
            edges3 = F.interpolate(edges, (128, 128))
            edges4 = F.interpolate(edges, (64, 64))
            edges5 = F.interpolate(edges, (32, 32))


            bound = tesnor_bound(gts, 3).cuda()
            bound2 = F.interpolate(bound, (256, 256))
            bound3 = F.interpolate(bound, (128, 128))

            out = model(images, tis)



            loss1 = IOUBCE(out[0], gts)
            loss2 = IOUBCE(out[1], gts2)
            loss3 = IOUBCE(out[2], gts3)
            loss4 = IOUBCE(out[3], gts4)
            loss5 = IOUBCE(out[4], gts5)

            loss11 = IOUBCE(out[6], edges)
            loss21 = IOUBCE(out[7], edges2)
            loss31 = IOUBCE(out[8], edges3)
            loss41 = IOUBCE(out[9], edges4)
            loss51 = IOUBCE(out[10], edges5)


            predict_bound0 = out[0]
            predict_bound1 = out[1]
            predict_bound2 = out[2]
            predict_bound0 = tesnor_bound(torch.sigmoid(predict_bound0), 3)
            predict_bound1 = tesnor_bound(torch.sigmoid(predict_bound1), 3)
            predict_bound2 = tesnor_bound(torch.sigmoid(predict_bound2), 3)
            loss6 = IOUBCEWithoutLogits(predict_bound0, bound)
            loss7 = IOUBCEWithoutLogits(predict_bound1, bound2)
            loss8 = IOUBCEWithoutLogits(predict_bound2, bound3)

            loss_ag = loss1 + loss11 + 0.8*(loss2 + loss21) + 0.6*(loss3 + loss31) + 0.4*(loss4 + loss41) + 0.2*(loss5 + loss51)
            loss_bound = loss6 + loss7 + loss8
            loss_trans = out[5]

            loss = loss_ag + loss_bound + loss_trans
            loss.backward()
            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_ag: {:.4f},'
                      'loss_bound: {:.4f},loss_trans: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(),
                             loss_ag.item(),loss_bound.item(), loss_trans.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_ag: {:.4f},'
                              'loss_bound: {:.4f},loss_trans: {:.4f} '.
                             format(epoch, opt.epoch, i, total_step, loss.item(),
                                    loss_ag.item(),loss_bound.item(), loss_trans.item()))
                writer.add_scalar('Loss', loss, global_step=step)

                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ground_truth', grid_image, step)
                grid_image = make_grid(bound[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/bound', grid_image, step)


                res = out[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/out', torch.tensor(res), step, dataformats='HW')
                res = predict_bound0[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/bound', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step

        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, tis, name, img_size = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            tis = tis.cuda()
            if opt.task == 'RGBD':
                tis = torch.cat((tis, tis, tis), dim=1)

            res = model(image, tis)

            res1 = F.interpolate(res[0], img_size, mode='bilinear', align_corners=True)
            gt = F.interpolate(gt, img_size, mode='bilinear', align_corners=True)
            res1 = torch.sigmoid(res1)
            res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)

            edge1 = F.interpolate(res[1], img_size, mode='bilinear', align_corners=True)
            edge1 = torch.sigmoid(edge1)
            edge1 = (edge1 - edge1.min()) / (edge1.max() - edge1.min() + 1e-8)

            mae_train = torch.sum(torch.abs(res1 - gt)) * 1.0 / (torch.numel(gt))

            mae_sum = mae_train.item() + mae_sum

        mae = mae_sum / test_loader.size

        writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} lastbestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('nextbestepoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
