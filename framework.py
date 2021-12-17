import torch
import torch.nn as nn
from torch.autograd import Variable as V
from sklearn.metrics import f1_score

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net = net().to(self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)

        self.loss = loss()
        self.loss_F1 = loss_F1()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        # img = V(torch.Tensor(img).cuda())
        img = V(torch.Tensor(img).to(self.device))

        mask = self.net.forward(img).squeeze(
        ).cpu().data.numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.to(self.device), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.to(self.device), volatile=volatile)

    def optimize(self, eval=False):
        self.forward()
        if not eval:
            self.optimizer.zero_grad()
            self.net.train()
            pred = self.net.forward(self.img)
        else:
            self.net.eval()
            pred = self.net.forward(self.img)
            F1 = self.compute_F1(pred, self.mask)
            
        loss = self.loss(self.mask, pred)
        if not eval:
            loss.backward()
            self.optimizer.step()
        return pred, loss.item()

    def compute_F1(pred, gt, args):
        """extract label list"""
        f1 = f1_score(pred.ravel(), np.array(gt).ravel())
        return f1

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
