"""
Main file for training Yolo model on Pascal VOC dataset
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import *

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 32 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 300
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "/home/cchun3/PROJECTS/YOLOv1/data/images"
LABEL_DIR = "/home/cchun3/PROJECTS/YOLOv1/data/labels"
Grid = 7
BBOX = 1
C_num = 20

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    mean_coloss = []
    mean_clloss = []
    mean_celoss = []
    mean_wloss = []
    mean_noloss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        # x, y = x_train.to(DEVICE), y_train.to(DEVICE)
        # out = model(x)
        loss, confidence_loss, class_loss, center_loss, window_loss, noobj_loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        mean_coloss.append(confidence_loss.item())
        mean_clloss.append(class_loss.item())
        mean_celoss.append(center_loss.item())
        mean_wloss.append(window_loss.item())
        mean_noloss.append(noobj_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    # print(f"Mean confidance loss was {sum(mean_coloss)/len(mean_coloss)}")
    # print(f"Mean class loss was {sum(mean_clloss)/len(mean_clloss)}")
    # print(f"Mean center loss was {sum(mean_celoss)/len(mean_celoss)}")
    # print(f"Mean window loss was {sum(mean_wloss)/len(mean_wloss)}")
    # print(f"Mean noobj loss was {sum(mean_noloss)/len(mean_noloss)}")

    return sum(mean_coloss)/len(mean_coloss), sum(mean_clloss)/len(mean_clloss), sum(mean_celoss)/len(mean_celoss), sum(mean_wloss)/len(mean_wloss), sum(mean_noloss)/len(mean_noloss)
def main_():
    model = Yolov1(split_size=Grid, num_boxes=BBOX, num_classes=C_num).to(DEVICE)
    model_opt = Yolov1(split_size=Grid, num_boxes=BBOX, num_classes=C_num).to(DEVICE)
    best_loss = 1e6
    # print(model)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    train_dataset = YoloDataSet(
        "data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = YoloDataSet(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=True,drop_last=True,)

    test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=True,drop_last=True,)
    test_class = []
    test_conf = []
    test_win = []
    test_noob = []
    test_cen = []

    train_class = []
    train_conf = []
    train_win = []
    train_noob = []
    train_cen = []
    

    for epoch in range(EPOCHS):
        print("epoch: " + str(epoch))
        mean_loss = []
        mean_coloss = []
        mean_clloss = []
        mean_celoss = []
        mean_wloss = []
        mean_noloss = []
        model.eval()
        for batch_idx, (x, y) in enumerate(test_loader):
          x, y = x.to(DEVICE), y.to(DEVICE)
          with torch.no_grad():
            out = model(x)
          # x, y = x_train.to(DEVICE), y_train.to(DEVICE)
          # out = model(x)
            loss, confidence_loss, class_loss, center_loss, window_loss, noobj_loss = loss_fn(out, y)
            mean_loss.append(loss.item())
            mean_coloss.append(confidence_loss.item())
            mean_clloss.append(class_loss.item())
            mean_celoss.append(center_loss.item())
            mean_wloss.append(window_loss.item())
            mean_noloss.append(noobj_loss.item())


        test_loss = sum(mean_loss)/len(mean_loss)
        if test_loss < best_loss:
          model_opt = model
          best_loss = test_loss
        model.train()
        print(f"test loss was {test_loss}")
        # print(f"Mean confidance loss was {sum(mean_coloss)/len(mean_coloss)}")
        # print(f"Mean class loss was {sum(mean_clloss)/len(mean_clloss)}")
        # print(f"Mean center loss was {sum(mean_celoss)/len(mean_celoss)}")
        # print(f"Mean window loss was {sum(mean_wloss)/len(mean_wloss)}")
        # print(f"Mean noobj loss was {sum(mean_noloss)/len(mean_noloss)}")
        test_class.append(sum(mean_clloss)/len(mean_clloss))
        test_conf.append(sum(mean_coloss)/len(mean_coloss))
        test_cen.append(sum(mean_celoss)/len(mean_celoss))
        test_win.append(sum(mean_wloss)/len(mean_wloss))
        test_noob.append(sum(mean_noloss)/len(mean_noloss))
        # pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, S=Grid, B=BBOX, C=C_num)

        # mean_avg_prec = mean_average_precision_built(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        # print(f"test mAP: {mean_avg_prec}")

        # pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, S=Grid, B=BBOX, C=C_num)

        # mean_avg_prec = mean_average_precision_built(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        # print(f"Train mAP: {mean_avg_prec}")

        conf_loss, cla_loss, cen_loss, win_loss, noo_los = train_fn(train_loader, model, optimizer, loss_fn)
        train_class.append(cla_loss)
        train_conf.append(conf_loss)
        train_cen.append(cen_loss)
        train_win.append(win_loss)
        train_noob.append(noo_los)
        ploLoss(epoch, train_class, test_class, "class")
        ploLoss(epoch, train_conf, test_conf, "confidence")
        ploLoss(epoch, train_win, test_win, "win")
        ploLoss(epoch, train_noob, test_noob, "noobj")
        ploLoss(epoch, train_cen, test_cen, "center")


    model.eval()
    torch.save(model_opt, PATH)

main_()

if __name__ == "__main__":
    main()