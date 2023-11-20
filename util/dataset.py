import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):    
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)      
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()#验证集总数1449.coco训练82081
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:#训练集[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]测试集 [1, 2, 3, 4, 5]
        sub_class_file_list[sub_c] = []
    if len(sub_list)>5:
        for l_idx in tqdm(range(len(list_read))):  # 5953张图片
            #forpascalvoc
            line = list_read[l_idx]  # 'JPEGImages/2008_000008.jpg SegmentationClassAug/2008_000008.png'
            line = line.strip()  #
            line_split = line.split(' ')
            image_name = os.path.join(data_root, line_split[
                0])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/JPEGImages/2008_000008.jpg'
            label_name = os.path.join(data_root, line_split[
                1])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/SegmentationClassAug/2008_000008.png'
            # line = list_read[l_idx]  # 'JPEGImages/2008_000008.jpg SegmentationClassAug/2008_000008.png'
            # line = line.strip()  #
            # line_split = line.split(' ')
            # image_name = os.path.join(data_root,'images/train2014',line_split[
            #     0][-29:])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/JPEGImages/2008_000008.jpg'
            # label_name = os.path.join(data_root, 'annotations/train2014',line_split[
            #     1][-29:])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/SegmentationClassAug/2008_000008.png'

            item = (image_name, label_name)
            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)  # (442, 500)
            label_class = np.unique(label).tolist()  # [0, 13, 15]

            if 0 in label_class:  # [0, 13, 15]
                label_class.remove(0)  # [13, 15],去掉背景
            if 255 in label_class:
                label_class.remove(255)  # [13, 15]
            # 先剔除小目标
            new_label_class = []
            for c in label_class:  # [13, 15]
                if c in sub_list:  ##[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                    tmp_label = np.zeros_like(label)  # (442, 500)
                    target_pix = np.where(label == c)  # 找到对应类的位置信息
                    tmp_label[target_pix[0], target_pix[1]] = 1  # 将对应位置设置为1
                    if tmp_label.sum() >= 2 * 32 * 32:  # 判断总的个数是否满足条件,相当于排除小目标
                        new_label_class.append(c)

            label_class = new_label_class  # [13, 15]
            # 在找当前训练集对应类的图片
            if len(label_class) > 0:  # 判断该图片是否有类满足上述条件,有就加入LIST中
                image_label_list.append(item)  # 共4760张
                for c in label_class:  # [13, 15]
                    if c in sub_list:  # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                        sub_class_file_list[c].append(item)  # 如果该类在训练类中,就加入训练集对应的类
    else:
        for l_idx in tqdm(range(len(list_read))):  # 5953张图片
            line = list_read[l_idx]  # 'JPEGImages/2008_000008.jpg SegmentationClassAug/2008_000008.png'
            line = line.strip()  #
            line_split = line.split(' ')
            # for pascal
            image_name = os.path.join(data_root, line_split[
                0])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/JPEGImages/2008_000008.jpg'
            label_name = os.path.join(data_root, 'SegmentationClassAug' + line_split[1][
                                                                          -16:])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/SegmentationClassAug/2008_000008.png'
            #for coco
            # image_name = os.path.join(data_root, 'images/train2014', line_split[
            #                                                              0][
            #                                                          -31:])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/JPEGImages/2008_000008.jpg'
            # label_name = os.path.join(data_root, 'annotations/train2014', line_split[
            #                                                                   1][
            #                                                               -31:])  # '/home/prlab/willow/Oneshot/data/VOCdevkit2012/VOC2012/SegmentationClassAug/2008_000008.png'

            item = (image_name, label_name)
            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)  # (442, 500)
            label_class = np.unique(label).tolist()  # [0, 13, 15]

            if 0 in label_class:  # [0, 13, 15]
                label_class.remove(0)  # [13, 15],去掉背景
            if 255 in label_class:
                label_class.remove(255)  # [13, 15]
            # 先剔除小目标
            new_label_class = []
            for c in label_class:  # [13, 15]
                if c in sub_list:  ##[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                    tmp_label = np.zeros_like(label)  # (442, 500)
                    target_pix = np.where(label == c)  # 找到对应类的位置信息
                    tmp_label[target_pix[0], target_pix[1]] = 1  # 将对应位置设置为1
                    if tmp_label.sum() >= 2 * 32 * 32:  # 判断总的个数是否满足条件,相当于排除小目标
                        new_label_class.append(c)

            label_class = new_label_class  # [13, 15]
            # 在找当前训练集对应类的图片
            if len(label_class) > 0:  # 判断该图片是否有类满足上述条件,有就加入LIST中
                image_label_list.append(item)  # 共4760张
                for c in label_class:  # [13, 15]
                    if c in sub_list:  # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                        sub_class_file_list[c].append(item)  # 如果该类在训练类中,就加入训练集对应的类
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list#对应总的满足条件的图片共4760张,sub_class_file_list对应的是一个字典,表示的是当前训练集所对应类的图片



class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   

        if not use_coco:
            self.class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))    

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 
        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          
        label_class = np.unique(label).tolist()#[0, 9, 11, 255]
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []       
        for c in label_class:#[9, 11]
            if c in self.sub_val_list:#[11, 12, 13, 14, 15]
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class#[11]
        assert len(label_class) > 0


        class_chosen = label_class[random.randint(1,len(label_class))-1]#11
        class_chosen = class_chosen
        target_pix = np.where(label == class_chosen)#11
        ignore_pix = np.where(label == 255)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255           


        file_class_chosen = self.sub_class_file_list[class_chosen]#11
        num_file = len(file_class_chosen)#第11类共含有73张图片

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1#从第11类的73张中选择一张作为支持集
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    
        
        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        else:
            return image, label, s_x, s_y, subcls_list, raw_label

