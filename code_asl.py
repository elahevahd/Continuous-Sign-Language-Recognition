import os
from os import listdir
from os.path import isfile, join
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import ast
import json
import multiprocessing
from multiprocessing import Pool
from PIL import Image
import csv
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import math
import time
import shutil
from videotransforms import video_transforms, volume_transforms
from code_matlab_cp2tform import get_similarity_transform_for_cv2
from code_dataloader import *
from code_error_gen import * 

##########################################################################
# transformaitons that used by full frame, face, and hands
# global variable used by all networks
##########################################################################
video_transform_list = video_transforms.Compose(
    [   video_transforms.Resize((136, 136)),
        video_transforms.CenterCrop((112, 112)),
        volume_transforms.ClipToTensor(channel_nb=3, div_255=True, numpy=False),
        video_transforms.Normalize((0.450, 0.422, 0.390), (0.151987, 0.14855, 0.1569))
    ]
    )
##########################################################################
# Camera Configuration
##########################################################################

def video_config_coordinates(coords_list):
    face_coords_dict={'Left_eye': list(range(36,42))+[68], 'Right_eye': list(range(42,48))+[69],'mouth': list(range(48,68))}
    Left_eye_list = [coords_list[index] for index in face_coords_dict['Left_eye']]
    Right_eye_list = [coords_list[index] for index in face_coords_dict['Right_eye']]
    mouth_list = [coords_list[index] for index in face_coords_dict['mouth']]
    Left_eye_c1 , Left_eye_c2 =  np.mean([item[0] for item in Left_eye_list ]) , np.mean([item[1] for item in Left_eye_list ])
    Left_eye_center = [Left_eye_c1 , Left_eye_c2]
    Right_eye_c1 , Right_eye_c2 =  np.mean([item[0] for item in Right_eye_list ]) , np.mean([item[1] for item in Right_eye_list ])
    Right_eye_center = [Right_eye_c1 , Right_eye_c2]
    eye_middle = np.mean([Left_eye_center,Right_eye_center],axis=0)
    mouth_c1 , mouth_c2 =  np.mean([item[0] for item in mouth_list ]) , np.mean([item[1] for item in mouth_list ])
    mouth_center = [mouth_c1 , mouth_c2]
    distance_dict = {}
    eye_mouth_distance = np.linalg.norm(eye_middle-np.array(mouth_center))
    eyes_distance = np.linalg.norm(np.array(Left_eye_center)-np.array(Right_eye_center))
    distance_dict['eyes_distance']=eyes_distance
    distance_dict['eye_mouth_distance']=eye_mouth_distance
    return distance_dict

def video_config_distance_from_camera(data_folder,too_far_threshold,too_close_threshold):
    print('\n\n################################################')
    print('video_config: check distance from camera')
    print('################################################') 
    result_list =[]
    Openpose_folder = data_folder + '/OpenPose/'
    if len(os.listdir(Openpose_folder))>0:
        openposename = sorted(os.listdir(Openpose_folder))[0].split('_000000000000')[0]
        end_frame = len(os.listdir(data_folder+'/OpenPose'))-1
        for frame_num in range(end_frame+1):
            openpose_path = Openpose_folder+openposename+'_'+'{:012d}'.format(frame_num)+'_keypoints.json'
            data = json.load(open(openpose_path))
            arr = data['people'][0]['face_keypoints_2d']
            coords_list=[(arr[i*3],arr[i*3+1]) for i in range(int(len(arr)/3))]
            distance_dict = video_config_coordinates(coords_list)
            eyes_distance=distance_dict['eyes_distance'] 
            eye_mouth_distance=distance_dict['eye_mouth_distance']
            if eyes_distance >0 and eye_mouth_distance>0:
                result_list.append([eyes_distance,eye_mouth_distance])
        if len(result_list)>0:
            avg_result = np.mean(result_list,axis=0)
            if avg_result[0] < too_far_threshold and avg_result[1]< too_far_threshold: 
                return [False , 'video_distance_too_far']
            elif avg_result[0] > too_close_threshold and avg_result[1]> too_close_threshold: 
                return [False , 'video_distance_too_close']
            else:
                return [True, 'good distance from camera']
        else:
            return [False, 'video not captured']
    else:
        return [False, 'video not captured']

def video_config_checking_light(frame_lst,t,min_color, max_color):
    print('\n\n################################################')
    print('video_config: check lighting')
    print('################################################') 
    color_list = [np.mean(frame) for frame in frame_lst[0:t]]
    if min(color_list) < min_color:
        return [False , 'video_light_too_dark']
    elif max(color_list) > max_color:
        return [False, 'video_light_too_bright']
    else:
        return [True , '']

####################################################################################################################################################
# FOLDERS
####################################################################################################################################################

def get_file_lst(vid_folder): 
    listOfFile = os.listdir(vid_folder)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(vid_folder, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_file_lst(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def get_new_video():
    print('\n\n################################################')
    print('STEP 1: Find the vidoes without CSV files')
    print('################################################')
    dropbox_vid_folder = '/home/ultra-server/Dropbox/NSF-LEARN/video/'    
    allFiles = get_file_lst(dropbox_vid_folder)
    vid_lst = [] 
    csv_lst = [] 
    name_id_dict = {} 
    for name in allFiles:
        idx = name.rfind('/')
        if name[-4:] == '.avi':
            vid_lst.append(name[idx+1:-4])
            name_id_dict[name[idx+1:-4]] = name
        elif name[-4:] == '.csv':
            csv_lst.append(name[idx+1:-13])
    vid_lst_without_csv = list(set(vid_lst) - set(csv_lst)) 
    if(len(vid_lst_without_csv) == 0):
        return None, None
    org_name = name_id_dict[vid_lst_without_csv[0]] 
    print('the video that need to be processed is: '+ vid_lst_without_csv[0]) 
    dest = shutil.copyfile(org_name, '/home/ultra-server/Documents/ASL/video_data/' + vid_lst_without_csv[0] + '.avi')
    return org_name, '/home/ultra-server/Documents/ASL/video_data/' + vid_lst_without_csv[0] + '.avi'

def create_folder(video_path):
    print('\n\n################################################')
    print('STEP 2: Create temperory folders for all the mid-data')
    print('################################################')
    data_folder = video_path.replace('video_data', 'tmp_data')
    data_folder = data_folder[:-4]
    Openpose_folder = data_folder + '/OpenPose/'
    frameHD_folder = data_folder + '/frames_HD/'   
    frameLD_folder = data_folder + '/frames_LD/'   
    Face_folder = data_folder + '/face/'
    WarpFace_folder = data_folder + '/warpped_face/'
    Lhand_folder = data_folder + '/Lhand/'
    Rhand_folder = data_folder + '/Rhand/'
    if not os.path.exists(Openpose_folder):
        os.makedirs(Openpose_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(frameHD_folder):
        os.makedirs(frameHD_folder)
    if not os.path.exists(frameLD_folder):
        os.makedirs(frameLD_folder)
    if not os.path.exists(Face_folder):
        os.makedirs(Face_folder)
    if not os.path.exists(WarpFace_folder):
        os.makedirs(WarpFace_folder)
    if not os.path.exists(Lhand_folder):
        os.makedirs(Lhand_folder)
    if not os.path.exists(Rhand_folder):
        os.makedirs(Rhand_folder)
    return data_folder

##########################################################################
# Read videos as a list of HD frames: frame size : 1080 * 1920 * 3
##########################################################################
def getFrameFromVideo(video_path, data_folder):
    print('\n\n################################################')
    print('STEP 3: Extract high resoltuion and low resolution frames from the video')
    print('################################################')
    hd_frame_folder = data_folder + '/frames_HD/'
    ld_frame_folder = data_folder + '/frames_LD/'
    cap = cv2.VideoCapture(video_path)
    frame_lst = []
    cnt = 0
    ret, frame = cap.read()
    while ret:
        frame_lst.append(frame)
        hd_img_name = hd_frame_folder + str(cnt) + '.png'
        ld_img_name = ld_frame_folder + str(cnt) + '.png'
        cv2.imwrite(hd_img_name, frame)
        ld_frame = cv2.resize(frame, (480, 270))
        ld_frame = ld_frame[15:255,120:360,:]
        cv2.imwrite(ld_img_name, ld_frame)
        cnt = cnt + 1
        ret, frame = cap.read()
    print('the number of frames: ' + str(cnt))
    return frame_lst

##########################################################################
# Openpose
##########################################################################
def OpenPose(video_path, data_folder):
    print('\n\n################################################')
    print('STEP 4: Running Openpose for all the high resolution frames of the video')
    print('################################################')
    Openpose_folder = data_folder + '/OpenPose/'
    command = './build/examples/openpose/openpose.bin -face -hand -display 0 --render_pose 0 --write_json {} --video {}'.format(Openpose_folder,video_path)
    os.system(command)

def return_coords(path, keywork):
    data = json.load(open(path))
    arr=data['people'][0][keywork]
    lst=[[arr[i*3],arr[i*3+1]] for i in range(int(len(arr)/3))]
    return lst

#########################################################################
# Crop data and register face from videos based on the coordinate from OpenPose
##########################################################################

def cropped_from_videos(saved_folder, frame_lst, cnt, file_name, coord_type, cropped_size):
        coords_list = []
        coords_list = return_coords(file_name, coord_type)
        x_min = min([item[0] for item in coords_list])
        x_max = max([item[0] for item in coords_list])
        y_min = min([item[1] for item in coords_list])
        y_max = max([item[1] for item in coords_list])
        c1,c2 = int(np.mean([x_min,x_max])), int(np.mean([y_min,y_max]))
        if (c1,c2) == (0,0):  
            c1, c2 = cropped_size , cropped_size
        face_coord = [c1-cropped_size,c1+cropped_size,c2-cropped_size,c2+cropped_size]  #x_min, x_max, y_min, y_max 
        cropped_face = frame_lst[cnt][face_coord[2]:face_coord[3],face_coord[0]:face_coord[1],:]
        img_name = saved_folder + str(cnt) + '.png'
        cv2.imwrite(img_name, cropped_face)

def get_face_hands_data(video_path, data_folder, frame_lst):
    print('\n\n################################################')
    print('STEP 5:  Get the cropped face, warpped face, and hand regions based on the resutls of Openpose')
    print('################################################')
    Openpose_folder = data_folder + '/OpenPose/'
    face_folder = data_folder + '/face/'
    Wface_folder = data_folder + '/warpped_face/'
    Lhand_folder = data_folder + '/Lhand/'
    Rhand_folder = data_folder + '/Rhand/'
    nframe =len(os.listdir(Openpose_folder))
    file_lst = [f for f in listdir(Openpose_folder) if isfile(join(Openpose_folder, f))]  
    file_lst.sort()
    face_size = 120 #240x240 images 
    hand_size = 80 #160x160 images 
    cnt = 0
    for keypointsF in file_lst:
        file_name = Openpose_folder + keypointsF  
        cropped_from_videos(face_folder, frame_lst, cnt, file_name, 'face_keypoints_2d', face_size)
        face_warpping(Wface_folder, frame_lst, cnt, file_name, 'face_keypoints_2d', face_size)
        cropped_from_videos(Lhand_folder, frame_lst, cnt, file_name, 'hand_left_keypoints_2d', hand_size)
        cropped_from_videos(Rhand_folder, frame_lst, cnt, file_name, 'hand_right_keypoints_2d', hand_size)
        cnt = cnt + 1  

##########################################################################
# Face Warping
##########################################################################
def alignment(src_img,src_pts):
    mean_face = [ [71,92],[71, 105], [72, 118], [74, 131], [79, 143], [86, 153], [96, 161], [108, 167], [121, 169], [133, 167], [144, 162], [154, 154], [161, 144], [165, 132], [167, 120], 
                [169, 108], [170, 95], [82, 77], [89, 73], [97, 71], [106, 72], [114, 75], [131, 76], [139, 73], [147, 73], [155, 76], [160, 81], [122, 87], [122, 94], [122, 100], [122, 107],
                [112, 116], [117, 118], [122, 119], [126, 118], [131, 117], [92, 89], [97, 86], [103, 86], [108, 89], [103, 91], [97, 91], [135, 91], [140, 88], [146, 88], [151, 91], [146, 93], [140, 92], 
                [104, 135], [110, 130], [116, 127], [121, 128], [126, 127], [133, 130], [138, 136], [133, 140], [126, 142], [121, 142], [115, 142], [109, 139], [107, 134], [116, 132], [121, 133], 
                [126, 133], [135, 135], [126, 135], [121, 135], [116, 135], [100, 88], [143, 90] ]
    crop_size = (240, 240)
    src_pts = np.array(src_pts).reshape(70,2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(mean_face).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    face_img = face_img[41: 169, 71:170, :]
    return face_img

def face_warpping(saved_folder, frame_lst, cnt, file_name, coord_type, cropped_size):
        coords_list = []
        coords_list = return_coords(file_name, coord_type)
        x_min = min([item[0] for item in coords_list])
        x_max = max([item[0] for item in coords_list])
        y_min = min([item[1] for item in coords_list])
        y_max = max([item[1] for item in coords_list])
        c1,c2 = int(np.mean([x_min,x_max])), int(np.mean([y_min,y_max]))
        if (c1,c2) == (0,0): #if the face is missed, don't register it! 
            cropped_face = frame_lst[cnt][0:240,0:240,:]
            target_img = cropped_face[41: 169, 71:170, :]
        else:
            face_coord = [c1-cropped_size,c1+cropped_size,c2-cropped_size,c2+cropped_size]  #x_min, x_max, y_min, y_max 
            for idx in range(len(coords_list)):
                coords_list[idx][0] = coords_list[idx][0] - (c1 - 120) #left upper corner x (x new origin)
                coords_list[idx][1] = coords_list[idx][1] - (c2 - 120) #left upper corner y (y new origin)
            cropped_face = frame_lst[cnt][face_coord[2]:face_coord[3],face_coord[0]:face_coord[1],:]
            target_img = alignment(cropped_face, coords_list)
        img_name = saved_folder + str(cnt) + '.png'
        cv2.imwrite(img_name, target_img)

##########################################################################
# Gesture, Face, Head processing
##########################################################################

def process_the_gestures(data_folder,start_frame,end_frame):
    print('\n\n################################################')
    print('STEP 6:  Process the Manual Gestures')
    print('################################################') 
    gesture_pred_list= gesture_testing(data_folder,start_frame,end_frame)
    gesture_window_list = window_generation(start_frame,end_frame,8,2)
    gesture_results = [gesture_pred_list[index]+[gesture_window_list[index]] for index in range(len(gesture_window_list))]
    gesture_processed_vote_dict = process_predictions(gesture_results) #majority vote: one prediction for each frame
    gesture_confidence_vote_dict = confidence_dict(gesture_results) #a dictionary for each frame: confidence score list for classes - > {'FINGERSPELLING': [58.71], 'QUESTION_WH': [39.92], 'Others': [36.55, 69.6]}
    return gesture_processed_vote_dict, gesture_confidence_vote_dict

def process_head_and_face(data_folder,start_frame,end_frame):
    print('\n\n################################################')
    print('STEP 7:  Process Head Movements and Facial Expressions')
    print('################################################') 
    face_pred_list , head_pred_list= face_head_testing(data_folder,start_frame, end_frame)  
    face_head_window_list = window_generation(start_frame,end_frame,32,2)
    face_results = [face_pred_list[index]+[face_head_window_list[index]] for index in range(len(face_head_window_list))]   
    head_results = [head_pred_list[index]+[face_head_window_list[index]] for index in range(len(face_head_window_list))]
    face_processed_vote_dict = process_predictions(face_results)
    face_confidence_vote_dict = confidence_dict(face_results)
    head_processed_vote_dict = process_predictions(head_results)
    head_confidence_vote_dict = confidence_dict(head_results)
    return face_processed_vote_dict, face_confidence_vote_dict, head_processed_vote_dict, head_confidence_vote_dict

def predictions_output(data_folder,start_frame,end_frame,lexical_temporal_distance=6):
    gesture_processed_vote_dict, gesture_confidence_vote_dict = process_the_gestures(data_folder,start_frame,end_frame)      
    face_processed_vote_dict, face_confidence_vote_dict, head_processed_vote_dict, head_confidence_vote_dict= process_head_and_face(data_folder,start_frame,end_frame)
    print('\n\n################################################')
    print('STEP 8: Gesture Face Head Based Predictions')
    print('################################################') 
    gesture_final_predictions = return_final_pred(gesture_processed_vote_dict, 'gesture')
    gesture_based_video_output=[]
    for gesture_class in gesture_final_predictions.keys():
        gesture_interval_list= gesture_final_predictions[gesture_class]
        for gesture_interval in gesture_interval_list:
            s,e = gesture_interval
            if e-s >3:
                expand_start = max(0,s-lexical_temporal_distance)
                expand_end = min(e+lexical_temporal_distance,end_frame-1)
                gesture_list=return_preds(gesture_processed_vote_dict,gesture_confidence_vote_dict,s,e)
                gesture_confidence = dict(gesture_list)[gesture_class]
                face_list =return_preds(face_processed_vote_dict,face_confidence_vote_dict,expand_start,expand_end)
                head_list = return_preds(head_processed_vote_dict,head_confidence_vote_dict,expand_start,expand_end)
                gesture_based_video_output.append([(s,e), (gesture_class,gesture_confidence), face_list, head_list])
    '-------------------------------------------------------------------------------------'
    face_final_predictions = return_final_pred(face_processed_vote_dict, 'face') 
    face_based_video_output=[]
    for face_class in face_final_predictions.keys():
        face_interval_list= face_final_predictions[face_class]
        for face_interval in face_interval_list:
            s,e = face_interval
            if e-s >3:
                face_list =return_preds(face_processed_vote_dict,face_confidence_vote_dict,s,e)
                face_confidence = dict(face_list)[face_class]
                head_list = return_preds(head_processed_vote_dict,head_confidence_vote_dict,s,e) #[('NEGATIVE', 99.97), ('YN-WH', 99.86)]
                face_based_video_output.append([face_interval, (face_class,face_confidence), head_list])
    '-------------------------------------------------------------------------------------'
    clause_list = gesture_final_predictions['clause_boundary']
    '-------------------------------------------------------------------------------------'
    return gesture_based_video_output, face_based_video_output, clause_list


def main():
    print('\n*************************** The Processing Begins ****************************************')
    while(True):

        start_time = time.time()
        org_name, video_path = get_new_video()
        if org_name == None:
            print('No Videos have been found')
            continue 

        data_folder = create_folder(video_path)
        frame_lst = getFrameFromVideo(video_path, data_folder) #returns a list of high resolution frames
        end_frame = len(frame_lst)-1
        start_frame  = 0
        OpenPose(video_path, data_folder) 
        get_face_hands_data(video_path, data_folder, frame_lst) 

        ##########################################################################
        # Video Configuration
        ##########################################################################
        minimum_duration = 50
        if len(frame_lst)< minimum_duration:
            print('video_camera_not_connected or video_was_too_short')
            camera_config_feedback('video_camera_not_connected or video_was_too_short',org_name)
            continue
        flag, message = video_config_checking_light(frame_lst,t=minimum_duration,min_color=135, max_color=160)
        if not flag:
            camera_config_feedback(message,org_name)
            continue       
        flag, message  = video_config_distance_from_camera(data_folder,too_far_threshold=30,too_close_threshold=50)
        if not flag:
            camera_config_feedback(message,org_name)
            continue

        ##########################################################################
        # Predictions and Feedback
        ##########################################################################
        gesture_based_video_output, face_based_video_output, clause_list= predictions_output(data_folder,start_frame,end_frame,lexical_temporal_distance=6)
        generate_error_file(gesture_based_video_output, face_based_video_output, clause_list,org_name)

        ##########################################################################
        torch.cuda.empty_cache()
        print('\n*************************** The Processing Ends ****************************************')
        print('The entire pipeline time consumtion: ' + str(time.time() - start_time))

if __name__ == '__main__':
    main()



