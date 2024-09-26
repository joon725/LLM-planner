# generating obj_info_dict_img : generating object information dictionary per image

import sys, os
sys.path.append('./BLIP')
import cv2
from environment_recognition_utils import ObjectDetection, BLIP_vfe, BLIP_tfe, obj_info_once
from camera import *
from ultralytics import YOLO
from BLIP.models.blip_vqa import blip_vqa
from BLIP.models.blip import blip_feature_extractor
from PIL import Image

import yaml
from munch import Munch
import argparse

def generate_obj_info_dict_img(object_detector,vfe,img,depth_frame,camera_matrix,camera_to_base_HT,base_to_world_HT):
    fx,fy,cx,cy = camera_matrix[0],camera_matrix[1],camera_matrix[2],camera_matrix[3]
    obj_info_dict_img = obj_info_once(object_detector,vfe,img)
    for obj in list(obj_info_dict_img.keys()):
        obj_xyxy = obj_info_dict_img[obj]['xyxy']
        obj_bb_center = obj_info_dict_img[obj]['bb_center']

        # xyxy 범위 내의 카메라 프레임 기준 포인트 클라우드
        x1,y1,x2,y2 = obj_xyxy[0],obj_xyxy[1],obj_xyxy[2],obj_xyxy[3]
        obj_xyxy_camera_pointclouds = convert_depth_to_pointclouds(depth_frame,x1,x2,y1,y2,fx,fy,cx,cy) # bbox의 xyxy영역의 픽셀들을 camera frame 기준의 3D 좌표들로 변환
        obj_base_pointclouds = transform_pointclouds(obj_xyxy_camera_pointclouds, camera_to_base_HT) # camera frame 기준의 3D 좌표들 -> robot base frame 기준의 3D 좌표들
        obj_world_pointclouds = transform_pointclouds(obj_base_pointclouds, base_to_world_HT) # robot base frame 기준의 3D 좌표들 -> world frame 기준의 3D 좌표들

        # bbox 중심의 카메라 프레임 기준 (x,y,z)
        bb_center_x = obj_bb_center[0]
        bb_center_y = obj_bb_center[1]
        bb_center_z = depth_frame[bb_center_y,bb_center_x]
        obj_bb_center_camera_coordinate = convert_depth_to_camera_frame(bb_center_x,bb_center_y,bb_center_z,fx,fy,cx,cy) # bbox의 중심 픽셀을 camera frame 기준의 3D 좌표로 변환
        obj_base_point = transform_point(obj_bb_center_camera_coordinate,camera_to_base_HT) # camera frame 3D 좌표 -> robot base frame 기준의 3D 좌표
        obj_world_point = transform_point(obj_base_point,base_to_world_HT) # robot base frame 3D 좌표 -> world frame 기준의 3D 좌표
        
        # world frame 기준, 각 물체의 bbox 안 픽셀에 해당하는 포인트 클라우드 및 각 물체의 bbox 중심 픽셀에 해당하는 좌표를 obj_info_img에 저장
        obj_info_dict_img[obj]['pointclouds'] = obj_world_pointclouds
        obj_info_dict_img[obj]['coordinate'] = obj_world_point    

    return obj_info_dict_img    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[SDC] Environment Recognition Module Test') 
    parser.add_argument('--config', default="./cfg/parameter.yaml", help='path to yaml config file', type=str) 
    parsed_args = parser.parse_args() 

    # add the settings in configuration file(yaml) to args.
    with open(parsed_args.config, 'r') as f: 
        params = yaml.load(f, Loader=yaml.FullLoader)     
    args = Munch(params, **vars(parsed_args)) 
    
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    print(f"current_file_path : {current_file_path}")
    print(f"current_dir : {current_dir}")

    # load YOLO
    model_det = YOLO(args.yolo_weight)
    print("[Object Detector Loaded]")

    # load BLIP-vfe
    model_VFE = blip_feature_extractor(pretrained='./weights/BLIP/model_base.pth', med_config = "./BLIP/configs/med_config.json", vit='base')
    print("[Visual Feature Extractor Loaded]")
    model_VFE.eval()
    model_VFE = model_VFE.to("cuda:0")

    # load BLIP-tfe
    model_TFE = blip_feature_extractor(pretrained='./weights/BLIP/model_base.pth', med_config = "./BLIP/configs/med_config.json", vit='base')
    print("[Text Feature Extractor Loaded]")
    model_TFE.eval()
    model_TFE = model_TFE.to("cuda:0")    

    # test img
    test_img_path = args.img_path
    test_img = cv2.imread(test_img_path) # bgr format
    print("[Test Image Loaded]")
    print(f"[INFO] Img shape : {test_img.shape}")

    # depth_frame
    depth_img_path = args.depth_path
    depth_frame = np.load(depth_img_path)    
    print("[Test Depth Frame Loaded]")
    print(f"[INFO] Depth Frame shape : {depth_frame.shape}")

    # camera parameter 
    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy

    camera_matrix = [fx,fy,cx,cy]

    # dummy HT matrix 
    camera_to_base_HT = np.identity(4)
    base_to_world_HT = np.identity(4) 

    # make obj_info_dict for one image
    obj_info_dict_img = generate_obj_info_dict_img(model_det,model_VFE,test_img,depth_frame,camera_matrix,camera_to_base_HT,base_to_world_HT)    
    print("[Object Info Dictionary for one Image Generated]")
    print(f"[INFO] obj_info_dict keys - {obj_info_dict_img.keys()}")
    obj_name = list(obj_info_dict_img.keys())[0]
    obj_keys = obj_info_dict_img[obj_name].keys()
    print(f"[INFO] obj_info_dict[object] keys - {obj_name} : {obj_keys}")

    print("[DONE]")
















