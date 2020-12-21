from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# Argument Parsing by class
class opt:
    image_folder = "data/samples"
    model_def = "config/yolov3.cfg"
    weights_path = "weights/yolov3.weights"
    class_path = "data/coco.names"
    conf_thres = 0.8
    nms_thres = 0.4
    batch_size = 1
    n_cpu = 0
    img_size = 416
    checkpoint_model= str()

def plot_one_box(x, img, color=1, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


# Model Load
print("Model Loading")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)

# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode

dataloader = DataLoader(
    ImageFolder(opt.image_folder, img_size=opt.img_size),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

# Webcam Define
print("Webcam Define")
width = 1280
height = 720

cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

frames = 0
start = time.time()

while True:
    ret_val, img = cam.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv reead img as BGR

    # Mirror 
    img = cv2.flip(img, 1)
    img_re = cv2.resize(img, (416, 416))
    
    input_imgs = transforms.ToTensor()(img_re)
    input_imgs = torch.unsqueeze(input_imgs, 0).to(device)

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        img_detections.extend(detections)
    
    # Create plot
    # Draw bounding boxes and labels of detections
    if detections[0] is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections[0], opt.img_size, img.shape[:2])

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            plot_one_box((x1,y1,x2,y2), img, label=classes[int(cls_pred)])
    
    frames += 1
    intv = time.time() - start
    if intv > 1:
        print("FPS of the video is {:5.2f}".format( frames / intv ))
        print(detections)
        start = time.time()
        frames = 0
    
    cv2.imshow('Demo webcam', img)
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
        
cam.release()
cv2.destroyAllWindows()