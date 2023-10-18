import os,csv
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.nets.gatector_1 import GaTectorBody
from lib.nets.yolo_training import YOLOLoss, weights_init
from lib.utils.callbacks import LossHistory
from lib.dataloader import GaTectorDataset, gatector_dataset_collate
from lib.utils.utils import get_anchors, get_classes
from lib.utils.utils_fit1 import fit_one_epoch
import cv2 
from PIL import Image
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
import time
import cv2
import numpy as np

classes_path = './data/anchors/voc_classes.txt'
anchors_path = './data/anchors/yolo_anchors.txt'
anchors_mask = [[0, 1, 2]]
class_names, num_classes = get_classes(classes_path)
model = GaTectorBody(anchors_mask, num_classes,1)


checkpoint = torch.load('data/86_04/86_04.pth')
model.load_state_dict(checkpoint)
model.eval()
with torch.no_grad():
    # path = '../dataset/images/COCO_train2014_000000000510.jpg'
    # image = Image.open('0002.jpg')
    # image = transforms.ToTensor()(transforms.Resize(240)(image)).unsqueeze(0)
    
    train_dataset = GaTectorDataset('./data/goo_dataset/gooreal/train_data/', './data/goo_dataset/gooreal/train.pickle', [224,224], num_classes, 0,train=True)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=8, pin_memory=True,
                         drop_last=True, collate_fn=gatector_dataset_collate)
    for iteration, batch in enumerate(gen):
            

            images, targets, faces, head, gaze_heatmap, gt_box = batch[0], batch[1], batch[2], batch[3], batch[4], \
                                                                 batch[7]
            
            
            
            images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
            
            
                
            faces = torch.from_numpy(faces).type(torch.FloatTensor).cuda()
            head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
            gaze_heatmap = torch.from_numpy(gaze_heatmap).type(torch.FloatTensor).cuda()
            targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
            
            heatmap = model.forward(images,faces,head,0)[0]
    
    
    print(heatmap.shape)

    heatmap_h, heatmap_w = heatmap.size(1), heatmap.size(2)

    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    temperature = 0.25
    tempered_heatmap = torch.log(torch.softmax(heatmap.view(-1), dim=0)) / temperature
    tempered_heatmap = torch.exp(tempered_heatmap) / torch.sum(torch.exp(tempered_heatmap))

    hm = tempered_heatmap.view(heatmap_h, heatmap_w)
    # hm[hm < hm.max() * 0.5] = 0
    plt.imshow(hm, cmap='Greys_r')
    plt.show()
    
    heatmap_img = cv2.resize(hm.numpy(), (heatmap_w*8, heatmap_h*8))
    heatmap_img = cv2.applyColorMap(((1-(heatmap_img / heatmap_img.max())) * 255).astype(np.uint8), cv2.COLORMAP_PARULA)
    weighted = cv2.addWeighted(heatmap_img, 0.7, (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), 0.3, 0)
    
    plt.imshow(weighted)
    plt.show()