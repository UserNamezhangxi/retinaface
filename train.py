import datetime
import math
import time

import torch
from torch import optim
from torch.utils import data

from data.data_augment import preproc
from data.wider_face import WiderFaceDetection, detection_collate
from layers.prior_box import PriorBox
from loss.multibox_loss import MultiBoxLoss
from models.retinaface import RetinaFace
from data.config import cfg_mnet


network = 'mobile0.25'
initial_lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
rgb_mean = (104, 117, 123) # bgr order
training_dataset = 'F:\\py\\Retinaface\\Pytorch_Retinaface\\data\\widerface\\train\\label.txt'
num_workers = 4
save_folder = "./weights/"
gamma = 0.1
num_classes = 2

if network == "mobile0.25":
    cfg = cfg_mnet

img_dim = cfg['image_size']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

# net to cuda
net = net.cuda()

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

criterion = MultiBoxLoss(num_classes, 0.35, 7)

with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()  # priors to cuda


def train():
    net.train()
    print('Loading Dataset...')
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch = 0
    start_iter = 0
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues: # 这里判断stepvalue 来调节 lr
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || loss_l: {:.4f} loss_c: {:.4f} loss_landm: {:.4f} || lr: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                .format(epoch, max_epoch,
                        (iteration % epoch_size) + 1, epoch_size,
                        iteration + 1, max_iter,
                        loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("adjust_learning_rate = {},epoch {}".format(lr, epoch))
    return lr

if __name__ == '__main__':
    train()