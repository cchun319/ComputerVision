"""
Implementation of Yolo Loss Function from the original yolo paper
"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
  def __init__(self, split = 7, boxes = 1, Classes = 20):
    super(YoloLoss, self).__init__()
    self.mse = nn.MSELoss(reduction= "sum")
    self.mae = nn.L1Loss()
    self.ce = nn.CrossEntropyLoss()
    self.S = split
    self.B = boxes
    self.C = Classes

    self.lambda_noobj = 1 # 0.5
    self.lambda_coord = 5 # 5
    self.lambda_confiance = 1 # 1
  def forward(self, pred, target):
    # shape of pred should be(N x (1470))
    predictions = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
    # N is the batch size
    iou_ = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) 

    # 0:19 class, 20:confidence, 21:22, center, 23:24, window size

    ### confidence loss ###
    label_confidence = target[..., 20] # N x 7 x 7 --> has object
    pred_confidence = predictions[...,20] # N x 7 x 7 
 
    confidence_loss = self.mse(label_confidence, label_confidence * pred_confidence) * self.lambda_confiance

    ### noobject loss ###
    noobj_id = 1 - label_confidence # 1: noobject, 0:otherwise # N x 7 x 7 --> no object
    nooob_target = noobj_id * label_confidence  # the value of the cell that should not detect object is kept, 

    nooobj_pred = noobj_id * pred_confidence 

    noobj_loss = self.lambda_noobj * self.mae(nooob_target, nooobj_pred)

    ### class loss ###
    label_class = target[..., 0:20]
    pred_class = predictions[..., 0:20]

    has_obj = torch.flatten(label_confidence.unsqueeze(-1) > 0)
    # print(has_obj.shape)
    label_class_id = torch.flatten(torch.argmax(label_class, dim=3)) #(Nx7x7, 1)
    # print(label_class_id.shape)
    label_class_id_filtered = label_class_id[has_obj == True]
    
    pred_list = torch.flatten(pred_class, end_dim = 2)
    pred_list_has_obj = pred_list[has_obj == True, :]

    class_loss = self.ce(pred_list_has_obj, label_class_id_filtered)

    # class_loss = self.mse(label_class, label_confidence.unsqueeze(-1) * pred_class) # punish the long rod that responsible for the class detection
    
    ### position/window_size loss ###
    label_center = target[..., 21:23]
    pred_center = predictions[...,21:23]

    center_loss = self.mse(label_center, label_confidence.unsqueeze(-1) * pred_center) * self.lambda_coord
    
    label_win_size = torch.sqrt(target[..., 23:25])
    pred_win_size = torch.sign(predictions[...,23:25]) * torch.sqrt(torch.abs(predictions[...,23:25]))

    window_loss = self.mse(label_win_size, label_confidence.unsqueeze(-1) * pred_win_size) * self.lambda_coord

    loss = confidence_loss + class_loss + center_loss + window_loss + noobj_loss

    return loss, confidence_loss, class_loss, center_loss, window_loss, noobj_loss