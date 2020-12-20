import torch
import torch.nn as nn
import tensorflow as tf
from collections import Counter
import numpy as np

import os
import pandas as pd
from PIL import Image

import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from model import Yolov1
from model import CNNLayer
from utils import *
from dataset import *

import cv2

DEVICE = "cuda" if torch.cuda.is_available else "cpu"

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

PATH2 = '/home/cchun3/Downloads/modelTue Dec 8 17 52 36 2020.pt'
model_ = Yolov1(split_size=7, num_boxes=1, num_classes=20).to(DEVICE)
model_ = torch.load(PATH2)
model_.eval()

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
# read the video in and seperate into frames
# transform to 448 and transform back to the original size
# use model to predict 
# draw bounding box and append 
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

label_dict = {6: "car", 8:"chair", 14:"human", 12:"horse", 7:"cat", 9:"ox", 5:"bus",1 :"bike", 11:"dog", 18:"train", 0:"airplane", 2:"bird", 19:"tv", 4:"bottle",13:"motorbike", 15:"plant",3: "boat",16:"sheep", 
17:"sofa", 10:"unknown"}

def Process_Video(vidcap):
  # video: video files
  # return : the video after identification
  success = True
  count = 0;

  # video = cv2.VideoWriter('filename.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (480, 480)) 
  while success and count < 10:
    success,image = vidcap.read()

    # name = './frames' + str(count) + '.jpg'  
    #     # writing the extracted images 
    # cv2.imwrite(name, image) 
  
        # increasing counter so that it will 
        # show how many frames are created 
    cv_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    
    im_pil = Image.fromarray(cv_img)

    image_tensor, _ = transform(im_pil, None)

    pred_s = model_(image_tensor.to(DEVICE).unsqueeze(0))

    boxes = cellboxes_to_boxes(pred_s)
    fboxes =  non_max_suppression(boxes[0], 0.5, 0.4, "midpoint")
    image_final_ = image_tensor.permute(1,2,0)
    plot_image(image_final_, fboxes, label_dict, count)

    # cv2.destroyAllWindows()

    count += 1
  # video.release()
  return None

video_path = "/home/cchun3/Downloads/test-v-trim_480.mp4"

def getOneItem(im_path, lb_path, S= 7, B = 1, C = 20):
  image = Image.open(im_path)
  true_label = []
  true_label2 = []
  with open(lb_path) as f:
    for label in f.readlines():
        class_label, x, y, width, height = [
          float(x) if float(x) != int(float(x)) else int(x)
          for x in label.replace("\n", "").split()
          ]
        true_label.append([class_label, x, y, width, height])
        true_label2.append([class_label, 1, x, y, width, height])

    # print(image.shape)
  image, _ = transform(image, None)
  label_matrix = torch.zeros((S, S, C + 5 * B))
  for box in true_label:
    class_label, x, y, width, height = box
    class_label = int(class_label)

    # i,j represents the cell row and cell column
    i, j = int(S * y), int(S * x) # 
    x_cell, y_cell = S * x, S * y


    width_cell, height_cell = width * S, height * S


    if label_matrix[i, j, 20] == 0:
      # Set that there exists an object
      label_matrix[i, j, 20] = 1

      # Box coordinates
      box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

      label_matrix[i, j, 21:25] = box_coordinates

      # Set one hot encoding for class_label
      label_matrix[i, j, class_label] = 1

  return image, label_matrix, true_label, true_label2

def printBox(boxes):
  for j in range(len(boxes)):
    print("predict: " + str(boxes[j][0]) + "\t | condidence: " + str("%.2f" % round(boxes[j][1], 2)) + "\t | x: " + str("%.2f" % round(boxes[j][2], 2)) + "\t | y: " + str("%.2f" % round(boxes[j][3],2)))



def main():
  vidcap = cv2.VideoCapture(video_path)
  processed_ = Process_Video(vidcap)
  # image = cv2.imread('./0.png',0)
  # print(image.shape)
  # test_dataset = VOCDataset("/home/cchun3/PROJECTS/YOLOv1/data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,)

  # test_loader = DataLoader(
  #       dataset=test_dataset,
  #       batch_size=1,
  #       num_workers=2,
  #       pin_memory=2,
  #       shuffle=True,
  #       drop_last=True,)

  # for img_, boxes in test_loader:
  #   pred_s = model_(img_.to(DEVICE))
  #   boxes_ = cellboxes_to_boxes(pred_s)
  #   fboxes =  non_max_suppression(boxes_[0], 0.5, 0.4, "midpoint")
  #   image_final_ = img_[0].permute(1,2,0)
  #   plot_image(image_final_, fboxes, label_dict)


  # test only one image with mAP

  # im_path = '/home/cchun3/PROJECTS/YOLOv1/data/images/000058.jpg'
  # lb_path = '/home/cchun3/PROJECTS/YOLOv1/data/labels/000058.txt'
  # image, true_cell, true_boxes, true_boxes_con = getOneItem(im_path, lb_path, S= 7, B = 1, C = 20)
  # y_list = []
  # y_list.append(true_cell)
  # x_list = []
  # x_list.append(image)

  # allpred, alltrue = get_bboxes_One(x_list, y_list, model_, 0.5, 0.4, pred_format="cells",box_format="midpoint",device="cuda",)
  
  # print("mAP: " + str(mean_average_precision_built(allpred, alltrue)))

  # pred_s = model_(image.to(DEVICE).unsqueeze(0))
  # boxes = cellboxes_to_boxes(pred_s)
  # fboxes = non_max_suppression(boxes[0], 0.5, 0.4, "midpoint")
  # fboxes.sort(key=lambda x:x[0])
  # true_boxes_con.sort(key=lambda x:x[0])
  # print("PREDICTIONBOX")
  # printBox(fboxes)
  # print("TRUEBOX")
  # printBox(true_boxes_con)
  # image = image.permute(1,2,0)
  # plot_image(image, fboxes, label_dict)
  # print("image shown")

  # save the processed_file
  return 0

if __name__ == "__main__":
    main()



  

  