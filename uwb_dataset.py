import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
#from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os 
import glob
import numpy as np
import random
import queue

import time
import cv2
from scipy import signal
from einops import rearrange, reduce, repeat
from PIL import Image

from collections import deque


class UWBDataset(Dataset):
    def __init__(self, args, mode='train'):
        
        '''
        dataset 처리
        rf와 이미지의 경우에는 init 할 때부터 읽어와서 메모리에 올리지만 gt는 데이터를 활용할 때마다 load함.
        mode - train : 학습을 위함.  rf, gt, img 다 있는 경우
                test : test를 위함. rf, gt, img 다 있는 경우 

        '''
        
        self.mode = mode

        self.load_cd = True if args.pose is not None else False
        if mode =='test' and args.pose is not None:
            self.load_cd = True
        self.load_hm = True if (args.pose == 'hm' or args.pose =='hmdr') else False
        self.load_img = args.vis and mode != 'train'
        self.load_feature = True if args.box_feature != 'x' else False


        self.print_once = True

        self.frame_stack_num = args.stack_num
        self.frame_skip = args.frame_skip
        self.num_txrx = 8
        self.stack_avg = args.stack_avg
        self.cutoff = args.cutoff
        assert self.frame_stack_num <= args.stack_avg 
        if self.stack_avg > 0:
            outlier_by_ewma = self.stack_avg + 1

        
        data_path = './sample_data'
        data_path_list = glob.glob(data_path + '/*')
        data_path_list = sorted(data_path_list)
        print("data_path = ", data_path_list)

        rf_data = []  
        raw_list = []
        target_list = []

        mask_list = [] 
        hm_list = []
        cd_list = []

        img_list = []
        filename_list =[]
        feature_list = []
        outlier_list = []
        remove_dir = []
        print("start - data read ", mode)
        
        train_dir = [0]
        test_dir = [0] 
        
        train_outlier_idx = []
        test_outlier_idx = []


        dir_count = 0 

        rf_index = -1  
        target_index = -1
        rf_frame = -1 

        hm_index = -1
        cd_index = -1

        img_index = -1
        filename_index = -1
        feature_index = -1

        frame_stack = deque(maxlen=self.stack_avg)
        not_stacked_list = []
        for file in data_path_list:
            if dir_count in remove_dir:
                dir_count += 1
                continue
            if mode == 'train' and dir_count not in train_dir:
                dir_count += 1
                continue
            elif mode == 'test' and dir_count not in test_dir:
                dir_count += 1
                continue

            if os.path.isdir(file) is True:
                rf_file_list = glob.glob(file + '/radar/*.npy')
                rf_file_list = sorted(rf_file_list)
                print('\n\n\tdir_count:', dir_count,'dir(raw):', file)
                print('\t# of data :', len(rf_file_list), "rf_idx =", rf_index)

                frame_stack.clear()
                dir_rf_index = -1

                for rf in rf_file_list:
                    rf_index += 1

                    if rf_index in outlier_list:
                        if mode=='train' and rf_index in train_outlier_idx:
                            print("train_outlier_idx = ", rf_index, rf)
                        
                        if mode=='test' and rf_index in test_outlier_idx:
                            print("test_outlier_idx = ", rf_index, rf)
                        dir_rf_index = -1
                        frame_stack.clear()
                        continue
                   
                    dir_rf_index += 1
                    raw_rf_load = np.load(rf)

                    temp_raw_rf = raw_rf_load[:, :, self.cutoff:]

                    if dir_rf_index == 1000:
                        print("dir_rf_index {} max, min, mean(raw) = ".format(dir_rf_index), np.max(raw_rf_load), np.min(raw_rf_load), np.mean(raw_rf_load))
                        print("dir_rf_index {} max, min, mean(input) = ".format(dir_rf_index), np.max(temp_raw_rf), np.min(temp_raw_rf), np.mean(temp_raw_rf), raw_rf_load.shape)
                                   
                    
                    temp_raw_rf = torch.tensor(temp_raw_rf).float()
                    temp_raw_rf = torch.flatten(temp_raw_rf, 0, 1)
                    raw_list.append(temp_raw_rf)
                    rf_frame +=1
                    frame_stack.append(rf_frame)

                    if len(frame_stack) == self.stack_avg and dir_rf_index >= outlier_by_ewma:
                        rf_data.append(list(frame_stack))
                    else:
                        not_stacked_list.append(rf_index)
                                        
                    if rf_index %10000 == 10 and len(frame_stack) == self.stack_avg:
                        print(f"rf_index {rf_index}, rf_frame {rf_frame}, rf_file = {rf}, frame_stack ={frame_stack[0]}~{frame_stack[-1]}[{self.stack_avg}]")
                        if self.frame_stack_num > 1 or self.stack_avg > 1:
                            tmp_rf = []
                            stack_idx = []
                            mean_rf = torch.zeros((self.num_txrx*self.num_txrx, 1024-self.cutoff))
                            for i in range(self.stack_avg):
                                if self.stack_avg > 1:
                                    raw_rf = raw_list[frame_stack[i]]
                                    mean_rf += raw_rf
                                    if i >= self.stack_avg - self.frame_stack_num and i%self.frame_skip == self.frame_skip-1:
                                        tmp_rf.append(raw_rf)
                                        stack_idx.append(i)
                            print("stack idx = ", stack_idx)
                            rf = torch.stack(tmp_rf, 0)
                            mean_rf /= self.stack_avg
                            mean_rf = mean_rf[None, :, :]
                            mean_rf = mean_rf.repeat(rf.shape[0], 1, 1)
                            print("mean ", mean_rf.shape, mean_rf[0,0,:3], torch.min(mean_rf), torch.max(mean_rf))
                            print("rf ", rf.shape, rf[0,0,:3], torch.min(rf), torch.max(rf))
                            rf -= mean_rf
                            print("rf ", rf.shape, rf[0,0,:3], torch.min(rf), torch.max(rf))


                if self.load_img:
                    img_file_list = glob.glob(file + '/image/*.jpg')
                    img_file_list = sorted(img_file_list)
                    print('\n\tdir(img):', file, '\t# of data :', len(img_file_list))

                    for img in img_file_list:
                        img_index += 1
                        filename_index += 1
                        if img_index in outlier_list or img_index in not_stacked_list:
                            continue
                        f_name = '{}/pred_feature/{}.npy'.format(file, img.split('/')[-1].split('.')[0])
                        
                        filename_list.append(f_name)
                        img_list.append(img)

                        if img_index %10000 == 10:
                            print(f"img_index {img_index} img_shape {cv2.imread(img).shape}")

                if self.load_hm:
                    hm_file_list = glob.glob(file + '/HEATMAP_MULTI64_mpii/*.npy')
                    hm_file_list = sorted(hm_file_list)
                    print('\n\tdir(posehm):', file, '\t# of data :', len(hm_file_list))
                
                    for hm in hm_file_list:
                        hm_index += 1
                        if hm_index in outlier_list or hm_index in not_stacked_list:
                            continue
                        
                        if hm_index %10000 == 10:
                            np_hm = np.load(hm)
                            print("hm_shape ", hm, np_hm.shape)
                                    
                        hm_list.append(hm)
                

                if self.load_cd:
                    cd_file_list = glob.glob(file + '/HEATMAP_COOR_mpii/*.npy')
                    cd_file_list = sorted(cd_file_list)
                    print('\n\tdir(pose_cd):', file, '\t# of data :', len(cd_file_list))
                
                    for cd in cd_file_list:
                        cd_index += 1
                        if cd_index in outlier_list or cd_index in not_stacked_list:
                            continue
                        np_cd = np.load(cd)
                        np_cd[:, :, 0] /= 640
                        np_cd[:, :, 1] /= 480
                                                        
                        cd_list.append(np_cd)

                        if cd_index %10000 == 10:
                            print("cd_shape ", cd, np_cd.shape, np_cd[0][0]) 


                target_file_list = glob.glob(file + '/box_people/*.npy')
                target_file_list = sorted(target_file_list)
                print('\n\tdir(target):', file + '/box_people/*.npy')
                print('\t# of data :', len(target_file_list))
                if len(target_file_list) == 0:
                    for _ in range(len(rf_file_list)):
                        target_index += 1
                        if target_index in outlier_list or target_index in not_stacked_list:
                            continue
                        target_list.append(None)
                else:
                    for target in target_file_list:
                        target_index += 1
                        if target_index in outlier_list or target_index in not_stacked_list:
                            continue
                        if target_index % 10000 == 1000:
                            print("target_shape ", target_index, target, np.load(target).shape, np.load(target))
                        target_list.append(np.load(target))


                
                if self.load_feature:
                    feature_file_list = glob.glob(file + '/imgfeature/*.npy')
                    feature_file_list = sorted(feature_file_list)
                    print('\n\tdir(feature16):', file, '\t# of data :', len(feature_file_list))

                    for feature in feature_file_list:
                        feature_index += 1
                        if feature_index in outlier_list or feature_index in not_stacked_list:
                            continue

                        if feature_index %10000 == 1000:
                            np_feature = np.load(feature)
                            print("featuremap_shape ", feature, np_feature.shape)
                            print(np_feature.max(), np_feature.min())


                        feature_list.append(feature)


            dir_count += 1

        self.rf_data = rf_data
        self.mask_list = mask_list
        self.hm_list = hm_list
        self.cd_list = cd_list
        self.img_list = img_list
        self.raw_list = raw_list
        self.feature_list = feature_list
        self.filename_list = filename_list

        self.target_list = target_list
        print(f"rf\t{len(rf_data)}/{outlier_by_ewma}\t raw\t{len(raw_list)}/{rf_frame}\t target\t{len(target_list)}")
        print(f"pose_cd\t{len(cd_list)}\t pose_hm\t{len(hm_list)}\t mask\t{len(mask_list)}")
        print(f"img\t{len(img_list)}\t file_name\t{len(filename_list)}")
        print(f"feature\t{len(feature_list)}")


        if self.mode =='train':
            if self.load_cd:
                assert len(rf_data) == len(cd_list)
            assert len(rf_data) == len(target_list)
        assert len(raw_list) == rf_frame + 1

        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        mask = np.ones((1))
        img = None
        f_name = None
        cd = None
        hm = None
        feature = None

        rf = self.get_rf(idx)
        target = self.target_list[idx] 

        if self.load_cd:
            cd = self.cd_list[idx]
        
        if self.load_hm:
            hm = self.get_hm(idx)
        
        if self.load_img:
            img = self.img_list[idx]
            img = cv2.imread(img)
            f_name = self.filename_list[idx]

        if self.load_feature:
            feature_name = self.feature_list[idx]
            feature = np.load(feature_name)

        
        if self.mode=='train':
            return rf, target, cd, hm, feature

        else:
            return rf, target, mask, cd, hm, idx, img, feature, f_name
            
       
    def get_rf(self, idx):
        rf = self.rf_data[idx]
        if self.stack_avg > 1:
            tmp_rf = []
            mean_rf = torch.zeros((self.num_txrx*self.num_txrx, 1024-self.cutoff))
            for i in range(self.stack_avg):
                raw_rf = self.raw_list[rf[i]]
                mean_rf += raw_rf
                if i >= self.stack_avg - self.frame_stack_num and i%self.frame_skip == self.frame_skip-1:
                    tmp_rf.append(raw_rf)
            rf = torch.stack(tuple(tmp_rf), 0)
            mean_rf /= self.stack_avg
            mean_rf = mean_rf[None, :, :]
            mean_rf = mean_rf.repeat(rf.shape[0], 1, 1)
            rf -= mean_rf
        else:
            rf = self.raw_list[rf[-1]].unsqueeze(0)
        
        return rf

    def get_hm(self, idx):
        pose = self.hm_list[idx]
        pose = np.load(pose)        
        return pose


def detection_collate(batch):
    rfs = []
    targets = []
    cds = []
    hms = []
    features = []


    for sample in batch:
        rf = sample[0].clone()
        target = torch.FloatTensor(sample[1]).clone()
        cd = torch.FloatTensor(sample[2]).clone() if sample[2] is not None else None
        hm = torch.FloatTensor(sample[3]).clone() if sample[3] is not None else None
        feature = torch.FloatTensor(sample[4][0]).clone() if sample[4] is not None else None
        
        rfs.append(rf)
        targets.append(target)
        cds.append(cd)
        hms.append(hm)
        features.append(feature)

    rfs = torch.stack(rfs)
 
    return rfs, targets, cds, hms, features

def detection_collate_val(batch):
    rfs = []
    targets = []
    masks = []
    cds =[]
    hms = []
    ids = []
    imgs = []
    features = []

    for sample in batch:
        rf = sample[0].clone()
        target = torch.FloatTensor(sample[1]).clone()
        mask = sample[2]
        cd = torch.FloatTensor(sample[3]).clone() if sample[3] is not None else None
        hm = torch.FloatTensor(sample[4]).clone() if sample[4] is not None else None
        idx = (sample[5], sample[8])
        img = sample[6]
        feature = torch.FloatTensor(sample[7][0]).clone() if sample[7] is not None else None

        rfs.append(rf)
        targets.append(target)
        ids.append(idx)
        masks.append(mask)
        cds.append(cd)
        hms.append(hm)
        imgs.append(img)
        features.append(feature)

    rfs = torch.stack(rfs)

    return rfs, targets, masks, cds, hms, ids, imgs, features


