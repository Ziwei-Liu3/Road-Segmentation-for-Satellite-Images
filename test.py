import torch
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time

from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet152, DinkNet34_less_pool


class TTAFrame():
    def __init__(self, net):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net().to(self.device)

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        return self.test_one_img_from_path_2(path)

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

    def load(self, path):
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(
                path, map_location=self.device))


def test():
    source_root = 'dataset/test_set_images'
    folder_names = sorted(os.listdir(source_root))
    img_names = [str(i)+'.png' for i in folder_names]
    solver = TTAFrame(DinkNet152)
    #  Load weights learned from training phase
    solver.load('weights/DinkNet152.th')
    tic = time()
    #  The path that stores the resulting mask of test set
    target = 'submits/DinkNet152/'
    os.mkdir(target)
    for i, name in enumerate(img_names):
        if (name == '.DS_Store'):
            continue
        if i % 10 == 0:
            print(i/10, '    ', '%.2f' % (time()-tic))
        print("Testing on ", name)
        path = os.path.join(source_root, folder_names[i], name)
        mask = solver.test_one_img_from_path(path)
        # Here we use threshold = 0.5 for each image, for 8 images batch, it is 4.0(mean)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        cv2.imwrite(target+name[:-4]+'_mask.png', mask.astype(np.uint8))
