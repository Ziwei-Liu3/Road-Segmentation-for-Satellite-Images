import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet152, DinkNet34_less_pool


# BATCHSIZE_PER_CARD = 8


class TTAFrame():
    def __init__(self, net):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net().to(self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        # if torch.cuda.is_available():
        #     batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        # else:
        #     batchsize = BATCHSIZE_PER_CARD
        return self.test_one_img_from_path_2(path)
        # if batchsize >= 8:
        #     return self.test_one_img_from_path_1(path)
        # elif batchsize >= 4:
        #     return self.test_one_img_from_path_2(path)
        # elif batchsize >= 2:
        #     return self.test_one_img_from_path_4(path)

    # def test_one_img_from_path_8(self, path):
    #     img = cv2.imread(path)  # .transpose(2,0,1)[None]
    #     img90 = np.array(np.rot90(img))
    #     img1 = np.concatenate([img[None], img90[None]])
    #     img2 = np.array(img1)[:, ::-1]
    #     img3 = np.array(img1)[:, :, ::-1]
    #     img4 = np.array(img2)[:, :, ::-1]

    #     img1 = img1.transpose(0, 3, 1, 2)
    #     img2 = img2.transpose(0, 3, 1, 2)
    #     img3 = img3.transpose(0, 3, 1, 2)
    #     img4 = img4.transpose(0, 3, 1, 2)

    #     img1 = V(torch.Tensor(np.array(img1, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))
    #     img2 = V(torch.Tensor(np.array(img2, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))
    #     img3 = V(torch.Tensor(np.array(img3, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))
    #     img4 = V(torch.Tensor(np.array(img4, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))

    #     maska = self.net.forward(img1).squeeze().cpu().data.numpy()
    #     maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
    #     maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
    #     maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

    #     mask1 = maska + maskb[:, ::-1] + \
    #         maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
    #     mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

    #     return mask2

    # def test_one_img_from_path_4(self, path):
    #     img = cv2.imread(path)  # .transpose(2,0,1)[None]
    #     img90 = np.array(np.rot90(img))
    #     img1 = np.concatenate([img[None], img90[None]])
    #     img2 = np.array(img1)[:, ::-1]
    #     img3 = np.array(img1)[:, :, ::-1]
    #     img4 = np.array(img2)[:, :, ::-1]

    #     img1 = img1.transpose(0, 3, 1, 2)
    #     img2 = img2.transpose(0, 3, 1, 2)
    #     img3 = img3.transpose(0, 3, 1, 2)
    #     img4 = img4.transpose(0, 3, 1, 2)

    #     img1 = V(torch.Tensor(np.array(img1, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))
    #     img2 = V(torch.Tensor(np.array(img2, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))
    #     img3 = V(torch.Tensor(np.array(img3, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))
    #     img4 = V(torch.Tensor(np.array(img4, np.float32) /
    #              255.0 * 3.2 - 1.6).to(self.device))

    #     maska = self.net.forward(img1).squeeze().cpu().data.numpy()
    #     maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
    #     maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
    #     maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

    #     mask1 = maska + maskb[:, ::-1] + \
    #         maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
    #     mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

    #     return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(self.device))
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).to(self.device))

        maska = self.net.forward(img5).squeeze(
        ).cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    # def test_one_img_from_path_1(self, path):
    #     img = cv2.imread(path)

    #     img90 = np.array(np.rot90(img))
    #     img1 = np.concatenate([img[None], img90[None]])
    #     img2 = np.array(img1)[:, ::-1]
    #     img3 = np.concatenate([img1, img2])
    #     img4 = np.array(img3)[:, :, ::-1]
    #     img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
    #     img5 = np.array(img5, np.float32)/255.0 * 3.2 - 1.6
    #     img5 = V(torch.Tensor(img5).to(self.device))

    #     mask = self.net.forward(img5).squeeze().cpu().data.numpy()
    #     mask1 = mask[:4] + mask[4:, :, ::-1]
    #     mask2 = mask1[:2] + mask1[2:, ::-1]
    #     mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

    #     return mask3

    def load(self, path):
        # self.net.load_state_dict(torch.load(path))
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(
                path, map_location=self.device))


if __name__ == '__main__':

    #source = 'dataset/test/'
    source_root = 'dataset/test_set_images'
    folder_names = sorted(os.listdir(source_root))
    # paths = []
    img_names = [str(i)+'.png' for i in folder_names]
    solver = TTAFrame(DinkNet152)
    solver.load('weights/DinkNet152_8_2e_4.th')
    tic = time()
    target = 'submits/DinkNet152_8_2e_4/'
    os.mkdir(target)
    for i, name in enumerate(img_names):
        if (name == '.DS_Store'):
            continue
        if i % 10 == 0:
            print(i/10, '    ', '%.2f' % (time()-tic))
        print(name)
        path = os.path.join(source_root, folder_names[i], name)
        mask = solver.test_one_img_from_path(path)
        # Here we use threshold = 0.5 for each image, for 8 images batch, it is 4.0(mean)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        # mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
        cv2.imwrite(target+name[:-4]+'_mask.png', mask.astype(np.uint8))
