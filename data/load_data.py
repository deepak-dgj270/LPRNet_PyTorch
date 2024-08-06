from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS =  [ 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-','a','b','c','d',
         'e','f','g','h','i','j','k','l','m','n','o','p','q',
         'r','s','t','u','v','w','x','y','z',' ','(',')'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            # Debug print to list directory contents
            print(f"Listing files in directory: {img_dir[i]}")
            print(os.listdir(img_dir[i]))
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
             #one_hot_base = np.zeros(len(CHARS))
             #one_hot_base[CHARS_DICT[c]] = 1
             label.append(CHARS_DICT[c])
        if len(label) < self.lpr_max_len:
            label.extend([CHARS_DICT['-']] * (self.lpr_max_len - len(label)))  # Padding with '-' (or any other pad character)
        elif len(label) > self.lpr_max_len:
            label = label[:self.lpr_max_len]     
        label_string = ''.join([CHARS[i] for i in label])    
        print(f"Image name: {imgname}, Label: {label}, Num Plate: {label_string}")    

        if len(label) == 7:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        #print("Test")
        pass