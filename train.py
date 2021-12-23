import torch
import random
import math


import os
import numpy as np

from time import time

from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet152, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data_preprocessing import ImageFolder

# SEED = 0


def train():

    # fix seed
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # The network need the size to be a multiple of 32, resize is intriduced
    ORIG_SHAPE = (400, 400)
    SHAPE = (384, 384)
    NAME = 'DinkNet152'
    BATCHSIZE_PER_CARD = 8

    # Loading the name of training images and groundtruth images
    train_root = 'dataset/train/'
    image_root = os.path.join(train_root, 'images')
    gt_root = os.path.join(train_root, 'groundtruth')
    image_list = np.array(sorted(
        [f for f in os.listdir(image_root) if f.endswith('.png')]))
    gt_list = np.array(sorted(
        [f for f in os.listdir(gt_root) if f.endswith('.png')]))

    # Randomly select 20% of training data for validation
    total_data_num = image_list.shape[0]
    validation_data_num = math.ceil(total_data_num * 0.2)
    validation_idx = random.sample(range(total_data_num), validation_data_num)
    new_train_indx = list(
        set(range(total_data_num)).difference(set(validation_idx)))

    val_img_list = image_list[validation_idx].tolist()
    val_gt_list = gt_list[validation_idx].tolist()
    image_list = image_list[new_train_indx].tolist()
    gt_list = gt_list[new_train_indx].tolist()

    solver = MyFrame(DinkNet152, dice_bce_loss, 2e-4)

    if torch.cuda.is_available():
        train_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        val_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    else:
        train_batchsize = BATCHSIZE_PER_CARD
        val_batchsize = BATCHSIZE_PER_CARD

    # Data preprocessing for training set
    train_dataset = ImageFolder(image_list, image_root, gt_root, SHAPE)
    # No data preprocessing for validation dataset
    val_dataset = ImageFolder(val_img_list, image_root, gt_root, SHAPE, False)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        num_workers=0)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batchsize,
        shuffle=True,
        num_workers=0)

    if not os.path.exists('logs/'):
        os.mkdir('logs/')

    mylog = open('logs/'+NAME+'.log', 'w')
    tic = time()
    no_optim = 0
    no_optim_valid = 0
    total_epoch = 300
    train_epoch_best_loss = 100.
    validation_epoch_best_loss = 100

    for epoch in range(1, total_epoch + 1):
        print('---------- Epoch:'+str(epoch) + ' ----------')
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        validation_epoch_loss = 0

        print('Train:')
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        # Writing log
        duration_of_epoch = int(time()-tic)
        mylog.write('********************' + '\n')
        mylog.write('--epoch:' + str(epoch) + '  --time:' + str(duration_of_epoch) + '  --train_loss:' + str(
            train_epoch_loss) + '\n')
        # Print training loss
        print('--epoch:', epoch, '  --time:', duration_of_epoch, '  --train_loss:',
              train_epoch_loss)

        #  Do validation every 5 epochs
        if epoch % 5 == 0:
            val_data_loader_iter = iter(val_data_loader)

            print("Validation: ")
            for val_img, val_mask in val_data_loader_iter:
                solver.set_input(val_img, val_mask)
                val_loss = solver.optimize(True)
                validation_epoch_loss += val_loss
            validation_epoch_loss /= len(val_data_loader_iter)
            # Writing log
            mylog.write('--epoch:' + str(epoch) +
                        '  --validation_loss:' + str(validation_epoch_loss) + '\n')
            # Print validation loss
            print('--epoch:', epoch,  '  --validation_loss:',
                  validation_epoch_loss)

            if validation_epoch_loss < validation_epoch_best_loss:
                no_optim_valid = 0
                validation_epoch_best_loss = validation_epoch_loss
                # Store the weight
                solver.save('weights/'+NAME+'.th')
            else:
                no_optim_valid += 1
                if no_optim_valid >= 3:
                    # Early Stop
                    mylog.write(
                        'Validation loss not improving, early stop at' + str(epoch)+'epoch')
                    print(
                        'Validation loss not improving, early stop at %d epoch' % epoch)
                    break

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss

        if no_optim > 6:
            # Early Stop
            mylog.write('early stop at' + str(epoch)+'epoch')
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/'+NAME+'.th')
            #  Adjust learning reate
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    mylog.write('Finish!')
    print('Finish!')
    mylog.close()
