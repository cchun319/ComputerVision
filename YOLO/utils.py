import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import cv2
import os
from PIL import Image


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # 49 x 6 
    assert type(bboxes) == list
    # print(bboxes)
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    # print(bboxes)
    while bboxes:
        chosen_box = bboxes.pop(0)
        bbox_temp = bboxes.copy()
        bboxes = []
        for box in bbox_temp: # not the same class or not overlap a lot 
          if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]),torch.tensor(box[2:]), box_format=box_format,) < iou_threshold:
            bboxes.append(box)

        bboxes_after_nms.append(chosen_box)
    # print("NMS: " + str(len(bboxes_after_nms)))
    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, class_dic, frame_n):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    print(im.shape)
    height, width, _ = im.shape

    # Create figure and axes
    # fig, ax = plt.subplots(1)
    # # Display the image
    # ax.imshow(im)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)
    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        class_ = int(box[0])
        confidence_ = box[1]
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )


        label_bbox = class_dic[class_] + ":::" + f"{100 * confidence_:.2f}" + "%"
        plt.text(upper_left_x * width, upper_left_y * height - 10, label_bbox, size=10, rotation=0,
         ha="left", va="bottom",
         bbox=dict(boxstyle="square",
                   ec=(1, 0, 0),
                   fc=(1, 0, 0),
                   )
         )
        
    
        # Add the patch to the Axes
        ax.add_patch(rect)
    if frame_n:
        plt.savefig(str(frame_n) + '.png', dpi=200, bbox_inches="tight", transparent=True, pad_inches=0)
    else:
        plt.show()
    

def get_bboxes(loader,model,iou_threshold,threshold,pred_format="cells",box_format="midpoint",device="cuda",):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx],iou_threshold=iou_threshold,threshold=threshold,box_format=box_format,)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7, B=1, C=20):
    """
    convert bounding box in grid size to original size
    return: boxes
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, C + 5 * B) 
    bounding_box = predictions[..., 21:25] # x, y , width, height # N x 7 x 7 x 4
    
    confidence = predictions[..., 20].unsqueeze(-1) # N x 7 x 7 

    pred_class = predictions[..., 0:20].argmax(-1).unsqueeze(-1) # N x 7 x 7

    # N x 49 x 6, return [pred_class, confidence, x, y, width, height]
    # print(torch.flatten(pred_class, 1, 2).shape)
    # print(torch.flatten(confidence, 1, 2).shape)
    # print(torch.flatten(bounding_box, 1, 2).shape)
    predict_boxes = torch.cat((torch.flatten(pred_class, 1, 2), torch.flatten(confidence, 1, 2), torch.flatten(bounding_box, 1, 2)), dim = -1)
    return predict_boxes

def cellboxes_to_boxes(out, S=7, B = 1, C = 20):
    predict_boxes = convert_cellboxes(out, S, B, C) # N x 49 x 6 
    
    batch_size = predict_boxes.shape[0]

    # predict_boxes_expand = torch.flatten(predict_boxes, 0, 1) # (N x 49, 1)
    # batch_id = torch.arange(batch_size).repeat(S*S,1)
    # batch_id_T = torch.transpose(batch_id, 1, 0)
    # batch_id_col = torch.reshape(torch.flatten(batch_id_T), (batch_size * 49, 1)) # (N x 49, 1) 
    # # print(batch_id_col.shape)
    # # print(predict_boxes_expand.shape)
    # all_bboxes_tensor = torch.cat((batch_id_col, predict_boxes_expand), dim= -1) # [id, predict_class, confidence, x, y, w, h] # (N x 49, 7)

    all_bboxes = predict_boxes.tolist()

    return all_bboxes

def mean_average_precision_built(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    predictions_class = []
    target_class = []
    ap = []

    for i in range(num_classes):
        predictions_class.append([])
        target_class.append([])
        ap.append(0.0)
    

    for pred in pred_boxes:
        c = int(pred[1])
        predictions_class[c].append(pred)
    
    for true_b in true_boxes:
        c = int(true_b[1])
        target_class[c].append(true_b)

    for cl_ in range(num_classes):
        ap_c = 0
        # print("classnum: " + str(len(target_class[cl_])))
        # print("prednum: " + str(len(predictions_class[cl_])))
        if len(target_class[cl_])!= 0 and len(predictions_class[cl_]) != 0:
            # print("class: " + str(cl_))
            ap_c = average_precision(predictions_class[cl_], target_class[cl_], iou_threshold)
        ap[cl_] = ap_c
    print("sumap: " + str(sum(ap)))

    return sum(ap) / len(ap)

def average_precision(predictions, ground_truth, iou_threshold):
  # return the average precision of a class  
  # predictions [train_idx, pred_class, confidence, x, y, w, h]
  predictions.sort(key=lambda x: x[2], reverse=True)
  
  TP_ = torch.zeros((len(predictions))) 
  FP_ = torch.zeros((len(predictions)))

  gt_used = torch.zeros((len(ground_truth)))
  for pred_id, pred in enumerate(predictions):
    # get the label in same image
    # print("prediction: " + str(pred))
    gt_ = []
    for bid, bbox in enumerate(ground_truth):
        if int(bbox[0]) == int(pred[0]) and int(gt_used[bid]) == 0:
            nbb = [bid] + bbox
            # print(nbb)
            gt_.append(nbb)

    if len(gt_) == 0:
      FP_[pred_id] = 1
      continue
    
    best_iou = 0
    best_id = 0
    for gt in gt_:
        iou_ = intersection_over_union(torch.tensor(pred[3:]), torch.tensor(gt[4:]), box_format="midpoint")
        # print("iou_: " + str(iou_))
        if iou_ > best_iou:
            best_iou = iou_
            best_id = gt[0]

    if best_iou >= iou_threshold and int(gt_used[best_id]) == 0: # ground truth haven't checked 
      TP_[pred_id] = 1
      gt_used[best_id] = 1
    else:
      FP_[pred_id] = 1
  
  TP_cumsum = torch.cumsum(TP_, dim=0)
  FP_cumsum = torch.cumsum(FP_, dim=0)
  rec = TP_cumsum / (len(ground_truth) + 1e-6)
  pre = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
  pre = torch.cat((torch.tensor([1]), pre))
  rec = torch.cat((torch.tensor([0]), rec))
  # print("gt_used: " + str(gt_used))
  # print("pre: " + str(pre))
  # print("rec: " + str(rec))
  auc = torch.trapz(pre, rec, dim = -1).item()
  # print("AP: " + str(auc))

  return auc

def getOneDataPoints(transform, img_dir, filename, label_dir, S, B, C):
  # 000058
  label_path = os.path.join(label_dir, filename + ".txt")
  boxes = []
  with open(label_path) as f:
      for label in f.readlines():
        class_label, x, y, width, height = [
          float(x) if float(x) != int(float(x)) else int(x)
          for x in label.replace("\n", "").split()
        ]

        boxes.append([class_label, x, y, width, height]) # 0 ~ 1
    
  img_path = os.path.join(img_dir, filename + ".jpg")
  image = Image.open(img_path)
  boxes = torch.tensor(boxes)

  if transform: # convert to 448 if the size of the image is not
    image, boxes = transform(image, boxes)

  label_m = torch.zeros((S, S, C + 5 * B)) # 7 x 7 x 30

  for box in boxes:
    class_label, x, y, width, height = box.tolist()
    class_label = int(class_label)

      # (x_center % grid_size) / grid_size <-> (x_center / 480) * grid_size 
    i, j = int(S * y), int(S * x) # the center of object in the grid -> the i,j grid is responsible for the detection 7x7 grid i, j // 
    x_cell, y_cell = S * x, S* y # x, y -> [0, 1] ->[0, 7]
      ### check encode and decode

    width_cell, height_cell = width * S, height * S # convert the width and length from image to grid scale
      ### 7 x 7 x 25, 0:19 class, 20, confidence, 21:25 x, y, w, h
    if label_m[i, j , 20] == 0:
      label_m[i, j , 20] = 1 # has object, probability density = 1,
      label_m[i, j , class_label] = 1 # indicate the class
      box_coor = torch.tensor([x_cell, y_cell, width_cell, height_cell])

      label_m[i, j, 21:25] = box_coor
  return image, label_m

def get_bboxes_One(x_train, y_train,model,iou_threshold,threshold,pred_format="cells",box_format="midpoint",device="cuda",):
    all_pred_boxes = []
    all_true_boxes = []
    # input list of training image [[image]], [7x7x25]
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    for i in range(len(x_train)):
        x = x_train[i].to(device)
        labels = y_train[i].to(device)

        with torch.no_grad():
            predictions = model(x.unsqueeze(0))

        true_bboxes = cellboxes_to_boxes(labels.unsqueeze(0))
        bboxes = cellboxes_to_boxes(predictions)

        nms_boxes = non_max_suppression(bboxes[0],iou_threshold=iou_threshold,threshold=threshold,box_format=box_format,)

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)

        for box in true_bboxes[0]:
                # many will get converted to 0 pred
            if box[1] > threshold:
                all_true_boxes.append([train_idx] + box)


    model.train()
    return all_pred_boxes, all_true_boxes

def ploLoss(time_, loss_train, loss_test, Class_name_):

  epochs = range(1, time_ + 2)

  plt.plot(epochs, loss_train, 'bo', label='Training loss')
  plt.plot(epochs, loss_test, 'b', label='Validation loss')
  plt.title(Class_name_ + ' Training and validation loss')
  plt.legend()

  plt.show()

def multipleTrainData(transform, IMG_DIR, num, LABEL_DIR, Grid, BBOX, C_num):
  images = []
  labels = []

  for i in range(1,num + 1):
    filename = str(i)
    filename = filename.zfill(6)
    # print(filename)
    x_, y_ = getOneDataPoints(transform, IMG_DIR, filename, LABEL_DIR, Grid, BBOX, C_num)
    images.append(x_)
    labels.append(y_)
  return images, labels