import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net = net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
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
        else:
            self.net.eval()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        if not eval:
            loss.backward()
            self.optimizer.step()
        return loss.item()

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
