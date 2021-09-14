import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from skimage.transform import resize
import os
import numpy as np

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    elif mode == 'test':
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False, mode='train', preprocessing=None):
        self.root_folder = root_folder
        self.mode = mode
        self.preprocessing = preprocessing
        self.transform = trans
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        elif self.mode == 'test':
            print("> Found %d label..." % (len(self.label_list)))
        
        self.cond = cond
        self.num_classes = 24
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        if self.img_list is not None:
            path = self.root_folder + "images/" + self.img_list[index]
            image = Image.open(path)
            image = np.asarray(image)

            image_size = 64
            if self.preprocessing:
                image = prepare_image(image, image_size)

            if self.transform != None:
                image = self.transform(image)

        label = self.label_list[index]
        
        return image, label
