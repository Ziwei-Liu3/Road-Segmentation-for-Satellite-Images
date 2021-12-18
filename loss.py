
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

import cv2
import numpy as np


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

class bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def __call__(self, y_true, y_pred):
        loss = self.bce_loss(y_pred, y_true)
        return loss

class F1_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(F1_Loss, self).__init__()

    def __call__(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        F1_approx = 2*(intersection + smooth)/(union + intersection + smooth)
                
        return 1 - F1_approx

# Jaccard Loss, refered from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class JaccLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccLoss, self).__init__()

    def __call__(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        Jacc = (intersection + smooth)/(union + smooth) + nn.BCELoss(inputs, targets)
                
        return 1 - Jacc