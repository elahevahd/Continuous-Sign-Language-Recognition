from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imp
import os
import collections
import json
import torch
import torchvision
import torchvision.transforms as transforms
from videotransforms import video_transforms, volume_transforms
import numpy as np
import PIL.Image as Image
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from torch.utils import data
from PIL import Image
import os
import os.path
import cv2
import PIL
from PIL import Image
# import h5py
import scipy
import random
import math
import ast
import itertools
from torch.autograd import Variable
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import random
from torch.utils import data
import torchvision
from resnet import resnet34, resnet10, resnet18, resnet50
from torch.optim import lr_scheduler
import itertools
from collections import Counter
from operator import itemgetter
from code_matlab_cp2tform import get_similarity_transform_for_cv2

################################################## Parameters ###########################################################################
numBatch = 64

################################################## NETWORKS ###########################################################################

# head_:  using all 29 coordinates from openpose  head_ratio = 134/240 + head images (not registered)
# face : face images (registered) + coordinates ( registered)
# gesture : Right hand and Left Hand images + Openpose Coordinates (21 points of Rhand + 21 points of Lhand + 3 points of center (Rhand, Lhand, face))   

gesture_cat_dict =  {'CONDITIONAL': 0, 'QUESTION_YN': 1, 'QUESTION_WH': 2, 'NEGATIVE': 3, 'Others': 4, 'TIME': 5, 'POINTING_PRONOUN': 6, 'clause_boundary': 7, 'FINGERSPELLING': 8}
gesture_label_dict =  {0: 'CONDITIONAL', 1:'QUESTION_YN', 2:'QUESTION_WH', 3:'NEGATIVE', 4:'Others', 5:'TIME', 6:'POINTING_PRONOUN', 7:'clause_boundary', 8:'FINGERSPELLING'}
gesture_all_classes= ['CONDITIONAL', 'QUESTION_YN', 'QUESTION_WH', 'NEGATIVE', 'Others', 'TIME', 'POINTING_PRONOUN', 'clause_boundary', 'FINGERSPELLING']
gesture_number_of_classes= len(set(list(gesture_cat_dict.values())))

face_cat_dict  = {'CONDITIONAL': 0, 'QUESTION_YN': 1, 'QUESTION_WH': 2, 'NEGATIVE': 3, 'Others': 4, 'RHQ': 5, 'TOPIC': 6}
face_label_dict=  {0: 'CONDITIONAL', 1:'QUESTION_YN', 2:'QUESTION_WH', 3:'NEGATIVE', 4:'Others', 5:'RHQ', 6:'TOPIC'}
face_all_classes= ['CONDITIONAL', 'QUESTION_YN', 'QUESTION_WH', 'NEGATIVE', 'Others', 'RHQ', 'TOPIC']
face_number_of_classes= len(set(list(face_cat_dict.values())))

head_cat_dict  = {'CONDITIONAL': 0, 'RHQ': 0, 'TOPIC': 0,'COND-RHQ-Topic':0,'QUESTION_YN': 1, 'QUESTION_WH': 1,  'YN-WH':1,'NEGATIVE': 2}
head_label_dict=  {0: 'COND-RHQ-Topic', 1:'YN-WH',  2:'NEGATIVE'}
head_all_classes= ['COND-RHQ-Topic', 'YN-WH', 'NEGATIVE']
head_number_of_classes= len(set(list(head_cat_dict.values())))

class HandsNet(nn.Module):
    def __init__(self):
        super(HandsNet, self).__init__()
        self.rhand_net = resnet34(num_classes =gesture_number_of_classes, shortcut_type = 'A', sample_size = 112, sample_duration = 8)
        self.lhand_net = resnet34(num_classes =gesture_number_of_classes, shortcut_type = 'A', sample_size = 112, sample_duration = 8)
        self.coords_fc_1 = nn.Linear(90, 40) 
        self.coords_fc_2 = nn.Linear(40, 20) 
        self.relu = nn.ReLU()
        self.BatchNorm1d = nn.BatchNorm1d(8, affine=False)
        self.fc = nn.Linear(512 * 2+160, 9)  
        print('--------------------        load the pretrained model        --------------------------------------')
        pretrain = torch.load('./pre_trained/kinetics/resnet-34-kinetics.pth')
        saved_state_dict = pretrain['state_dict']
        print('----------------------------------------------------------')
        hand_new_params = self.rhand_net.state_dict().copy()
        for name, param in hand_new_params.items():
            if 'module.'+name in saved_state_dict and param.size() == saved_state_dict['module.'+name].size():
                hand_new_params[name].copy_(saved_state_dict['module.'+name])
                print('copying ' + name + '  from   ' + 'module.'+name)
        self.rhand_net.load_state_dict(hand_new_params)
        print('----------------------------------------------------------')
        hand_new_params = self.lhand_net.state_dict().copy()
        for name, param in hand_new_params.items():
            if 'module.'+name in saved_state_dict and param.size() == saved_state_dict['module.'+name].size():
                hand_new_params[name].copy_(saved_state_dict['module.'+name])
                print('copying ' + name + '  from   ' + 'module.'+name)
        self.lhand_net.load_state_dict(hand_new_params)
        print('----------------------------------------------------------')

    def forward(self, rhand_clip, lhand_clip,coords_clip ):
        rhand_feat = self.rhand_net(rhand_clip)
        lhand_feat = self.lhand_net(lhand_clip)
        coords_feat = self.coords_fc_1(coords_clip)
        coords_feat = self.BatchNorm1d(coords_feat)
        coords_feat = self.relu(coords_feat)
        coords_feat = self.coords_fc_2(coords_feat)
        coords_feat = self.BatchNorm1d(coords_feat)
        coords_feat = self.relu(coords_feat)
        batch= coords_feat.size()[0]
        coords_feat=coords_feat.view(batch,1,160)  
        coords_feat = torch.squeeze(coords_feat,1)
        feat = torch.cat((rhand_feat, lhand_feat,coords_feat), dim =1)
        feat = self.fc(feat)
        return feat

class FaceNet(nn.Module): 
    def __init__(self):
        super(FaceNet, self).__init__()
        self.face_net = resnet34(num_classes = face_number_of_classes, shortcut_type = 'A', sample_size = 112, sample_duration = 32)
        self.fc_layer1= nn.Linear(106, 20)
        self.fc_layer2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        self.LeakyRelU = nn.LeakyReLU(0.1)
        self.BatchNorm1d_1 = nn.BatchNorm1d(32, affine=False)
        self.BatchNorm1d_2 = nn.BatchNorm1d(20, affine=False)
        self.fc = nn.Linear(512+200, face_number_of_classes)  

        print('--------------------        load the pretrained model        --------------------------------------')
        pretrain = torch.load('./pre_trained/kinetics/resnet-34-kinetics.pth')
        saved_state_dict = pretrain['state_dict']
        new_params = self.face_net.state_dict().copy()
        for name, param in new_params.items():
            if 'module.'+name in saved_state_dict and param.size() == saved_state_dict['module.'+name].size():
                new_params[name].copy_(saved_state_dict['module.'+name])
        self.face_net.load_state_dict(new_params)
        print('----------------------------------------------------------')

    def forward(self, video_clip,coords_clip ):
        face_feat = self.face_net(video_clip)
        coords_feat= self.fc_layer1(coords_clip) 
        coords_feat = self.BatchNorm1d_1(coords_feat)
        coords_feat = self.relu(coords_feat)
        coords_feat= coords_feat.permute(0, 2, 1)
        coords_feat=self.fc_layer2(coords_feat) 
        coords_feat = self.BatchNorm1d_2(coords_feat)
        coords_feat = self.relu(coords_feat)
        batch= coords_feat.size()[0]
        coords_feat=coords_feat.view(batch,1,200)  
        coords_feat = torch.squeeze(coords_feat,1)
        feat = torch.cat((face_feat,coords_feat), dim =1)
        feat = self.fc(feat)
        return feat


class headNet(nn.Module):
    def __init__(self):
        super(headNet, self).__init__()
        self.head_net = resnet34(num_classes = head_number_of_classes, shortcut_type = 'A', sample_size = 112, sample_duration = 32)
        self.fc_layer1= nn.Linear(58, 20)
        self.fc_layer2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        self.LeakyRelU = nn.LeakyReLU(0.1)
        self.BatchNorm1d_1 = nn.BatchNorm1d(32, affine=False)
        self.BatchNorm1d_2 = nn.BatchNorm1d(20, affine=False)
        self.fc = nn.Linear(512+200, head_number_of_classes)  

        print('--------------------        load the pretrained model        --------------------------------------')
        pretrain = torch.load('./pre_trained/kinetics/resnet-34-kinetics.pth')
        saved_state_dict = pretrain['state_dict']
        new_params = self.head_net.state_dict().copy()
        for name, param in new_params.items():
            if 'module.'+name in saved_state_dict and param.size() == saved_state_dict['module.'+name].size():
                new_params[name].copy_(saved_state_dict['module.'+name])
        self.head_net.load_state_dict(new_params)
        print('----------------------------------------------------------')

    def forward(self, video_clip,coords_clip ):
        head_feat = self.head_net(video_clip)
        coords_feat= self.fc_layer1(coords_clip) 
        coords_feat = self.BatchNorm1d_1(coords_feat)
        coords_feat = self.relu(coords_feat)
        coords_feat= coords_feat.permute(0, 2, 1)
        coords_feat=self.fc_layer2(coords_feat) 
        coords_feat = self.BatchNorm1d_2(coords_feat)
        coords_feat = self.relu(coords_feat)
        batch= coords_feat.size()[0]
        coords_feat=coords_feat.view(batch,1,200)  
        coords_feat = torch.squeeze(coords_feat,1)
        feat = torch.cat((head_feat,coords_feat), dim =1)
        feat = self.fc(feat)
        return feat

################################################## GET THE COORDINATES ###########################################################################
# Head coordinates : contour ,nose, eyes centers, mouth centers
head_coords_dict={
'contour': list(range(17)), 
'nose': list(range(27,36)), 
'Left_eye': list(range(36,42))+[68], 'Right_eye': list(range(42,48))+[69],
'Left_brow': list(range(17,22)) , 'Right_brow': list(range(22,27)), 
'mouth': list(range(48,68))
}
# face coordinates : nose, eyes , brows, mouth
face_coords_dict={
'nose': list(range(27,36)), 
'Left_eye': list(range(36,42))+[68], 'Right_eye': list(range(42,48))+[69],
'Left_brow': list(range(17,22)) , 'Right_brow': list(range(22,27)), 
'mouth': list(range(48,68))
}
#---------------------------------------------GET HEAD COORDINATES------------------------------------------------------------------

def return_head_coords(frame_num,openpose_path,t=120):
    openpose_path = openpose_path+'_'+'{:012d}'.format(frame_num)+'_keypoints.json'
    data = json.load(open(openpose_path))
    ###################################################################################################
    arr = data['people'][0]['face_keypoints_2d']
    coords_list=[(arr[i*3],arr[i*3+1]) for i in range(int(len(arr)/3))]
    #################################
    contour_list = [coords_list[index] for index in head_coords_dict['contour']]
    nose_list = [coords_list[index] for index in head_coords_dict['nose']]
    Left_eye_list = [coords_list[index] for index in head_coords_dict['Left_eye']]
    Right_eye_list = [coords_list[index] for index in head_coords_dict['Right_eye']]
    mouth_list = [coords_list[index] for index in head_coords_dict['mouth']]
    #################################
    Left_eye_c1 , Left_eye_c2 =  np.mean([item[0] for item in Left_eye_list ]) , np.mean([item[1] for item in Left_eye_list ])
    Left_eye_center = [(Left_eye_c1 , Left_eye_c2)]
    Right_eye_c1 , Right_eye_c2 =  np.mean([item[0] for item in Right_eye_list ]) , np.mean([item[1] for item in Right_eye_list ])
    Right_eye_center = [(Right_eye_c1 , Right_eye_c2)]
    mouth_c1 , mouth_c2 =  np.mean([item[0] for item in mouth_list ]) , np.mean([item[1] for item in mouth_list ])
    mouth_center = [(mouth_c1 , mouth_c2)]
    #################################
    l = [contour_list, nose_list, Left_eye_center, Right_eye_center, mouth_center]
    head_coords=(list(itertools.chain.from_iterable(l)))
    #################################
    bb_x_min = min([item[0] for item in coords_list])
    bb_x_max = max([item[0] for item in coords_list])
    bb_y_min = min([item[1] for item in coords_list])
    bb_y_max = max([item[1] for item in coords_list])
    c1,c2 = int(np.mean([bb_x_min,bb_x_max])), int(np.mean([bb_y_min,bb_y_max]))
    ###################################################################################################
    x_org,y_org = [c1-t,c2-t] 
    shifted_coords_list = [(x-x_org,y-y_org) for (x,y) in head_coords]
    head_ratio = 134/240 
    # scaled_coords_list = [(round(x*head_ratio,2), round(y*head_ratio,2)) for (x,y) in shifted_coords_list]
    scaled_coords_list = [((x*134)/240, (y*134)/240 ) for (x,y) in shifted_coords_list]

    x_list=[item[0] for item in scaled_coords_list]
    y_list=[item[1] for item in scaled_coords_list]
    x_mean , x_std , y_mean , y_std = [67.54, 17.52, 70.24, 14.90]
    x_list= [(item-x_mean)/x_std for item in x_list]
    y_list= [(item-y_mean)/y_std for item in y_list]
    normalized_frame_coords = np.array(x_list+y_list)
    return normalized_frame_coords

#---------------------------------------------- GET REGISTERED FACE COORDINATES---------------------------------------------------------------

def return_registeredface_coords(frame_num,openpose_path, t=120):
    openpose_path = openpose_path+'_'+'{:012d}'.format(frame_num)+'_keypoints.json'
    data = json.load(open(openpose_path))
    arr = data['people'][0]['face_keypoints_2d']
    coords_list=[(arr[i*3],arr[i*3+1]) for i in range(int(len(arr)/3))]
    ###################################################################################################
    nose_list = [coords_list[index] for index in face_coords_dict['nose']]
    Left_eye_list = [coords_list[index] for index in face_coords_dict['Left_eye']]
    Right_eye_list = [coords_list[index] for index in face_coords_dict['Right_eye']]
    Left_brow_list = [coords_list[index] for index in face_coords_dict['Left_brow']]
    Right_brow_list = [coords_list[index] for index in face_coords_dict['Right_brow']]
    mouth_list = [coords_list[index] for index in face_coords_dict['mouth']]
    #################################
    l = [nose_list, Left_eye_list, Right_eye_list,Left_brow_list,Right_brow_list, mouth_list]
    face_coords=(list(itertools.chain.from_iterable(l)))
    #################################
    bb_x_min = min([item[0] for item in coords_list])
    bb_x_max = max([item[0] for item in coords_list])
    bb_y_min = min([item[1] for item in coords_list])
    bb_y_max = max([item[1] for item in coords_list])
    c1,c2 = int(np.mean([bb_x_min,bb_x_max])), int(np.mean([bb_y_min,bb_y_max]))
    ###################################################################################################
    x_org,y_org = [c1-t,c2-t] 
    shifted_coords_list = [(x-x_org,y-y_org) for (x,y) in face_coords]
    coord_arr = np.array(shifted_coords_list).astype(np.float32)  #240x240
    n = 53 #coord_arr,sahpe[0]
    #################################
    mean_coords_list = [(122, 87), (122, 94), (122, 100), (122, 107), (112, 116), (117, 118), (122, 119), (126, 118), (131, 117), (92, 89), (97, 86), (103, 86), (108, 89), (103, 91), (97, 91), (100, 88), (135, 91), (140, 88), (146, 88), (151, 91), (146, 93), (140, 92), (143, 90), (82, 77), (89, 73), (97, 71), (106, 72), (114, 75), (131, 76), (139, 73), (147, 73), (155, 76), (160, 81), (104, 135), (110, 130), (116, 127), (121, 128), (126, 127), (133, 130), (138, 136), (133, 140), (126, 142), (121, 142), (115, 142), (109, 139), (107, 134), (116, 132), (121, 133), (126, 133), (135, 135), (126, 135), (121, 135), (116, 135)]
    mean_arr = np.array(mean_coords_list).astype(np.float32)
    #################################
    tfm = get_similarity_transform_for_cv2(coord_arr,mean_arr)
    #################################
    registered_coords=[]
    for index in range(n):
        a =np.array(list(coord_arr[index,:])+[1])
        des_point=list(np.matmul(tfm,a))        
        registered_coords.append(des_point)
    registered_coords = [(x-71, y-41) for (x,y) in registered_coords]
    #################################
    x_list=[item[0] for item in registered_coords]
    y_list=[item[1] for item in registered_coords]
    #################################
    x_mean , x_std , y_mean , y_std = [50.45, 18.17, 65.91, 25.0]
    x_list= [(item-x_mean)/x_std for item in x_list]
    y_list= [(item-y_mean)/y_std for item in y_list]
    normalized_frame_coords = np.array(x_list+y_list)
    return normalized_frame_coords

#-----------------------------------------------GET HANDS COORDINATES----------------------------------------------------------------

def organ_return_centers(data, keyword):
    arr = data['people'][0][keyword]
    coords_list=[(round(arr[i*3]),round(arr[i*3+1])) for i in range(int(len(arr)/3))]
    bb_x_min = min([item[0] for item in coords_list])
    bb_x_max = max([item[0] for item in coords_list])
    bb_y_min = min([item[1] for item in coords_list])
    bb_y_max = max([item[1] for item in coords_list])
    c1,c2 = int(np.mean([bb_x_min,bb_x_max])), int(np.mean([bb_y_min,bb_y_max]))
    return [c1,c2, coords_list]

def return_hands_coords(frame_num,openpose_path, t=80):
    openpose_path = openpose_path+'_'+'{:012d}'.format(frame_num)+'_keypoints.json'
    data = json.load(open(openpose_path))
    ###################################################################
    centers=[]
    c1,c2,coords_list  = organ_return_centers(data, 'hand_left_keypoints_2d')        
    centers.append((c1,c2))
    x_org,y_org = [c1-t,c2-t] 
    shifted_coords_list = [(x-x_org,y-y_org) for (x,y) in coords_list]
    left_scaled_coords = [(x*134/160, y*134/160) for (x,y) in shifted_coords_list]
    ###################################################################
    c1,c2,coords_list = organ_return_centers(data, 'hand_right_keypoints_2d')
    centers.append((c1,c2))
    x_org,y_org = [c1-t,c2-t] 
    shifted_coords_list = [(x-x_org,y-y_org) for (x,y) in coords_list]
    right_scaled_coords = [(x*134/160, y*134/160) for (x,y) in shifted_coords_list]
    ###################################################################
    c1,c2,coords_list = organ_return_centers(data, 'face_keypoints_2d')
    centers.append((c1,c2))
    ###################################################################
    centers = [(x*480/1920,y*270/1080) for (x,y) in centers]
    frame_coords = left_scaled_coords + right_scaled_coords + centers
    x_list=[item[0] for item in frame_coords]
    y_list=[item[1] for item in frame_coords]
    ###################################################################
    x_mean , x_std , y_mean , y_std = [78.83, 47.63, 76.33, 35.95]
    x_list= [(item-x_mean)/x_std for item in x_list]
    y_list= [(item-y_mean)/y_std for item in y_list]
    normalized_frame_coords = np.array(x_list+y_list)
    ###################################################################
    return normalized_frame_coords


############################################################################################################################################

def return_openpose_path(data_folder):
    Openpose_folder = data_folder + '/OpenPose/'
    openposename = sorted(os.listdir(Openpose_folder))[0].split('_000000000000')[0]
    openpose_path = Openpose_folder+openposename
    return openpose_path

video_transform_list = video_transforms.Compose([
    video_transforms.Resize((134,134)),
    video_transforms.CenterCrop((112, 112)),
    volume_transforms.ClipToTensor(),
    video_transforms.Normalize((0.450, 0.422, 0.390), (0.151987, 0.14855, 0.1569))
])


def hands_loader(data_folder, start_frame,end_frame): #hands of entire video.
    Rh_video_path = data_folder + '/Rhand/'
    Lh_video_path = data_folder +'/Lhand/'
    openpose_path = return_openpose_path(data_folder)
    rhand_clip=[]
    lhand_clip=[]
    hands_coords_clip=[]
    for frame_num in range(start_frame,end_frame+1):
        img_name = '{}.png'.format(frame_num)
        ######################################################
        Rh_img_path = Rh_video_path + img_name
        rhand_img = Image.open(Rh_img_path).convert('RGB')
        rhand_clip.append(rhand_img)
        ######################################################
        Lh_img_path = Lh_video_path + img_name
        lhand_img = Image.open(Lh_img_path).convert('RGB')
        lhand_clip.append(lhand_img)
        ######################################################
        normalized_hands_coords = return_hands_coords(frame_num,openpose_path)
        hands_coords_clip.append(normalized_hands_coords)
    ######################################################
    rhand_clip= video_transform_list(rhand_clip)
    lhand_clip= video_transform_list(lhand_clip)
    hands_coords_clip= torch.FloatTensor(hands_coords_clip)
    ######################################################
    return rhand_clip, lhand_clip, hands_coords_clip



def face_head_loader(data_folder, start_frame, end_frame,temporal_duration=32):
    Regface_video_path = data_folder + '/warpped_face/'
    head_video_path = data_folder + '/face/'
    openpose_path = return_openpose_path(data_folder)
    ######################################################
    head_clip=[]
    head_coords_clip=[]
    registered_face_clip=[]
    registeredface_coords_clip=[]
    for frame_num in range(start_frame,end_frame+1):
        img_name = '{}.png'.format(frame_num)
        ######################################################
        head_img_path= head_video_path+img_name
        head_img = Image.open(head_img_path).convert('RGB')
        head_clip.append(head_img)
        ######################################################
        normalized_head_coords = return_head_coords(frame_num,openpose_path)
        head_coords_clip.append(normalized_head_coords)
        ######################################################
        Regface_img_path = Regface_video_path + img_name
        Regface_img = Image.open(Regface_img_path).convert('RGB')
        registered_face_clip.append(Regface_img)
        ######################################################
        normalized_registeredface_coords = return_registeredface_coords(frame_num,openpose_path)
        registeredface_coords_clip.append(normalized_registeredface_coords)
    ######################################################
    registered_face_clip = video_transform_list(registered_face_clip)
    head_clip = video_transform_list(head_clip)
    registeredface_coords_clip=torch.FloatTensor(registeredface_coords_clip)
    head_coords_clip = torch.FloatTensor(head_coords_clip)
    ######################################################
    return head_clip, head_coords_clip, registered_face_clip, registeredface_coords_clip


###################################TESTING #######################################################################################################
def window_generation(start_frame,end_frame, window_len,stride):
    window_list=[]
    threshold= window_len-1
    if start_frame+threshold >= end_frame:
        window_list.append([start_frame,end_frame])
    else:
        i=0
        while  start_frame+i*stride+threshold < end_frame :
            S, E = start_frame+i*stride , start_frame+i*stride+window_len
            E= min(E,end_frame)
            window_list.append([S,E])
            i = i+1
    return window_list

def group_sliding_window(window_list):
    batch_list=[]
    if len(window_list) < numBatch:
        batch_list = [window_list]
    else:
        end_window_index = len(window_list)-1
        i=0
        while  i*numBatch <= end_window_index :
            S, E = i*numBatch ,(i+1)*numBatch
            if E <= end_window_index:
                batch_list.append(window_list[S:E])
            else:
                batch_list.append(window_list[S:end_window_index+1])
            i = i+1
    return batch_list

def face_head_testing(data_folder, start_frame, end_frame):
    with torch.no_grad():
        face_net = torch.load('./models_updated/face_net.pkl')   
        face_net = face_net.cuda()
        face_net.eval()
        # face_net = torch.nn.DataParallel(face_net)
        # cudnn.benchmark = True
        # face_net = face_net.to('cuda')
        head_net = torch.load('./models_updated/head_net.pkl')       
        head_net = head_net.cuda()
        head_net.eval()
        ################################################################
        head_clip, head_coords_clip, registered_face_clip, registeredface_coords_clip = face_head_loader(data_folder, start_frame, end_frame)
        head_clip = head_clip.unsqueeze(0)
        head_coords_clip = head_coords_clip.unsqueeze(0)
        registered_face_clip = registered_face_clip.unsqueeze(0)
        registeredface_coords_clip = registeredface_coords_clip.unsqueeze(0)
        ################################################################
        face_head_window_list = window_generation(start_frame,end_frame,32,2)
        grouped_face_head_list = group_sliding_window(face_head_window_list)
        ################################################################
        face_confidence_list =[]
        face_prediction_list =[]
        head_confidence_list =[]
        head_prediction_list =[]
        for grouped_face_head in grouped_face_head_list: #[[384, 416], [386, 418], [388, 420]]
            s,e = grouped_face_head[0]
            stacked_head_clip = head_clip[:,:,s:e,:,:]         
            stacked_registered_face_clip = registered_face_clip[:,:,s:e,:,:]        
            stacked_head_coords_clip= head_coords_clip[:,s:e,:]     
            stacked_registeredface_coords_clip= registeredface_coords_clip[:,s:e,:]     
            for window_index in range(1,len(grouped_face_head)):
                s,e= grouped_face_head[window_index]
                stacked_head_clip = torch.cat((stacked_head_clip, head_clip[:,:,s:e,:,:]))
                stacked_registered_face_clip = torch.cat((stacked_registered_face_clip, registered_face_clip[:,:,s:e,:,:]))
                stacked_head_coords_clip = torch.cat((stacked_head_coords_clip, head_coords_clip[:,s:e,:]))
                stacked_registeredface_coords_clip = torch.cat((stacked_registeredface_coords_clip, registeredface_coords_clip[:,s:e,:]))
            stacked_head_clip= Variable(stacked_head_clip).cuda()  
            stacked_registered_face_clip= Variable(stacked_registered_face_clip).cuda()  
            stacked_head_coords_clip= Variable(stacked_head_coords_clip).cuda()  
            stacked_registeredface_coords_clip= Variable(stacked_registeredface_coords_clip).cuda()  
            #############################################################
            pred = face_net(stacked_registered_face_clip,stacked_registeredface_coords_clip) #[3, 3, 32, 112, 112] ,[3, 32, 106]
            sm = torch.nn.Softmax() #FaceNet(nn.Module)
            probabilities = sm(pred) 
            _, p = pred.topk(1, 1, True)
            class_index_list = p.t()[0].tolist()
            face_confidence_list =face_confidence_list+ [round(probabilities[index,class_index_list[index]].item()*100,2) for index in range(len(class_index_list))]
            face_prediction_list =face_prediction_list+ [face_label_dict[class_index_list[index]] for index in range(len(class_index_list))]
            #############################################################
            pred = head_net(stacked_head_clip,stacked_head_coords_clip) #[3, 3, 32, 112, 112] , [3, 32, 58] 
            sm = torch.nn.Softmax()
            probabilities = sm(pred) 
            _, p = pred.topk(1, 1, True)
            class_index_list = p.t()[0].tolist()
            head_confidence_list =head_confidence_list+ [round(probabilities[index,class_index_list[index]].item()*100,2) for index in range(len(class_index_list))]
            head_prediction_list =head_prediction_list+ [head_label_dict[class_index_list[index]] for index in range(len(class_index_list))]
    ################################################################
    face_pred_list = [[face_prediction_list[index],face_confidence_list[index]] for index in range(len(face_confidence_list))]
    head_pred_list = [[head_prediction_list[index],head_confidence_list[index]] for index in range(len(head_confidence_list))]
    ################################################################
    return face_pred_list , head_pred_list

def gesture_testing(data_folder,start_frame,end_frame):  
    with torch.no_grad():
        net = torch.load('./models_updated/hands_net.pkl')       
        net = net.cuda()
        net.eval()
        ##########################################
        rhand_clip, lhand_clip, hands_coords_clip = hands_loader(data_folder,start_frame,end_frame)
        rhand_clip = rhand_clip.unsqueeze(0)#torch.Size([1,3, 421, 112, 112])
        lhand_clip = lhand_clip.unsqueeze(0)#torch.Size([1,3, 421, 112, 112])
        hands_coords_clip = hands_coords_clip.unsqueeze(0)#torch.Size([1,421, 90]
        ##########################################
        gesture_window_list = window_generation(start_frame,end_frame,8,2)
        grouped_gesture_list = group_sliding_window(gesture_window_list)
        ##########################################
        confidence_list =[]
        gesture_prediction_list =[]
        for grouped_gesture in grouped_gesture_list:
            s,e = grouped_gesture[0]
            stacked_rhand_clip = rhand_clip[:,:,s:e,:,:]         # stacked_rhand_clip.size()
            stacked_lhand_clip = lhand_clip[:,:,s:e,:,:]         # stacked_lhand_clip.size()
            stacked_coords_clip= hands_coords_clip[:,s:e,:]      # stacked_coords_clip.size()
            for window_index in range(1,len(grouped_gesture)):
                s,e= grouped_gesture[window_index]
                stacked_rhand_clip = torch.cat((stacked_rhand_clip, rhand_clip[:,:,s:e,:,:] ))
                stacked_lhand_clip = torch.cat((stacked_lhand_clip, lhand_clip[:,:,s:e,:,:] ))
                stacked_coords_clip = torch.cat((stacked_coords_clip, hands_coords_clip[:,s:e,:] ))
            stacked_rhand_clip= Variable(stacked_rhand_clip).cuda()  
            stacked_lhand_clip= Variable(stacked_lhand_clip).cuda()  
            stacked_coords_clip= Variable(stacked_coords_clip).cuda() 
            pred = net(stacked_rhand_clip,stacked_lhand_clip,stacked_coords_clip)
            sm = torch.nn.Softmax()
            probabilities = sm(pred) 
            _, p = pred.topk(1, 1, True)
            class_index_list = p.t()[0].tolist()
            confidence_list =confidence_list+ [round(probabilities[index,class_index_list[index]].item()*100,2) for index in range(len(class_index_list))]
            gesture_prediction_list =gesture_prediction_list+ [gesture_label_dict[class_index_list[index]] for index in range(len(class_index_list))]
    ##########################################
    gesture_pred_list = [[gesture_prediction_list[index],confidence_list[index]] for index in range(len(confidence_list))]
    #[['clause_boundary',0.3],.. ]
    return gesture_pred_list



################################################## POST PROCESSING ######################################################################

def confidence_dict(list_results):
    v_dict={}
    for pred_instance in list_results:
        class_label = pred_instance[0]
        confidence_score = pred_instance[1]
        start, end = pred_instance[2]
        for frame_num in range(start,end+1):
            if frame_num not in v_dict.keys():
                v_dict[frame_num]={}
            if class_label not in v_dict[frame_num].keys():
                v_dict[frame_num][class_label]=[]
            if class_label in v_dict[frame_num].keys():
                v_dict[frame_num][class_label].append(confidence_score)
    return v_dict


def process_predictions(list_results): #majority vote #assigns exactly one prediction to each frame 
    vote_dict={}
    for pred_instance in list_results:
        class_label = pred_instance[0]
        start, end = pred_instance[2]
        for frame_num in range(start,end+1):
            if frame_num not in vote_dict.keys():
                vote_dict[frame_num]={}
            if class_label in vote_dict[frame_num].keys():
                vote_dict[frame_num][class_label]+=1
            if class_label not in vote_dict[frame_num].keys():
                vote_dict[frame_num][class_label]=1

    frame_list = sorted(vote_dict.keys())
    processed_vote_dict={}
    for frame_num in frame_list:
        sorted_predictions = sorted([(key,vote_dict[frame_num][key]) for key in vote_dict[frame_num].keys()], key = itemgetter(1), reverse=True)
        highest_freq = sorted_predictions[0][1]
        highfreq_pred_list = [key for key in vote_dict[frame_num].keys() if vote_dict[frame_num][key]==highest_freq]
        if len(highfreq_pred_list)==1:   # => frame_num zero always fall here!
            processed_vote_dict[frame_num] = highfreq_pred_list[0]
        else:
            if processed_vote_dict[frame_num-1] in highfreq_pred_list:
                processed_vote_dict[frame_num] = processed_vote_dict[frame_num-1]
            else:
                processed_vote_dict[frame_num]=highfreq_pred_list[0]
    return processed_vote_dict



def segment_finder(processed_vote_dict,class_type): #find the intervals for each class
    frame_list = sorted(processed_vote_dict.keys())
    frame_array = [1]*len(frame_list)
    for frame_num in frame_list:
        if processed_vote_dict[frame_num] == class_type:
            frame_array[frame_num]=0   
    ######################################
    start_list=[]
    for index in range(len(frame_array)):
        value = frame_array[index]
        if index ==0 and value ==0:
            start_list.append(index)
        if index >0 and value ==0:
            if frame_array[index-1]==1:
                start_list.append(index)
    ######################################
    segment_list=[]
    for start_index in start_list:
        current_index=start_index
        while current_index < len(frame_array) and frame_array[current_index]==0:
            current_index +=1 
        end_index= current_index-1
        segment_list.append([start_index,end_index])
    return segment_list

def return_final_pred(processed_vote_dict, keyword):
    if keyword == 'gesture':
        class_list = ['CONDITIONAL','QUESTION_YN','QUESTION_WH','NEGATIVE','clause_boundary','TIME']
    if keyword == 'face':
        class_list = ['CONDITIONAL','QUESTION_YN','QUESTION_WH','NEGATIVE','TOPIC']
    if keyword =='head':
        class_list = ['COND-RHQ-Topic','YN-WH','NEGATIVE']
    # d= dict([(frame_num,processed_vote_dict[frame_num][0]) for frame_num in processed_vote_dict.keys()])
    d = processed_vote_dict
    final_predictions={}
    for class_type in class_list:
        final_predictions[class_type] = segment_finder(d,class_type)
    return final_predictions

def return_preds(processed_vote_dict,confidence_vote_dict,start,end):
    d={}
    for frame_num in range(start,end+1):
        predicted_label = processed_vote_dict[frame_num]
        confidence_score = max(confidence_vote_dict[frame_num][predicted_label])
        if predicted_label not in d.keys():
            d[predicted_label]=[]
        d[predicted_label].append(confidence_score)
    res_list=[(key,max(d[key])) for key in d.keys()]
    return res_list

