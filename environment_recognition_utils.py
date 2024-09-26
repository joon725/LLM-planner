# Yolo
from PIL import Image
import torch
import math
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2

class ObjectDetection():
    def __init__(self,model,img,save=False,save_txt=False,conf=0.7):
        self.model = model
        self.img = img
        self.save = save
        self.save_txt = save_txt
        self.conf = conf
        self.detected_obj_name_raw_list = []
        self.detected_obj_name_list = []
        self.box_center_list = []
        self.box_coord_list = []
        self.detected_obj_name_list_coord = {}
        self.results = None
    def _get_names(self):
        names = self.model.names
        return names
    def _detect(self):
        self.results = self.model.predict(source=self.img, save=self.save, save_txt=self.save_txt, conf=self.conf)
        return self.results
    def _get_box_center_coordinate(self):
        xyxys = self.results[0].boxes.xyxy
        object_num = xyxys.shape[0]
        # print(f"object_num : {object_num}")
        for i in range(object_num):
            x_center = int((xyxys[i][0] + xyxys[i][2])/2)
            y_center = int((xyxys[i][1] + xyxys[i][3])/2)
            center = (x_center,y_center)
            self.box_center_list.append(center)
        return self.box_center_list   
    def _get_box_coordinates(self):
        xyxys = self.results[0].boxes.xyxy
        object_num = xyxys.shape[0]
        for i in range(object_num):
            box_coords = (int(xyxys[i][0]),int(xyxys[i][1]),int(xyxys[i][2]),int(xyxys[i][3]))
            self.box_coord_list.append(box_coords)
        return self.box_coord_list   
    def _get_detected_objs_raw(self):
        classes = self._get_names()
        for r in self.results:
            for c in r.boxes.cls:
                self.detected_obj_name_raw_list.append(classes[int(c)])
        return self.detected_obj_name_raw_list         
    def _get_detected_objs(self):
        classes = self._get_names()
        for r in self.results:
            for c in r.boxes.cls:
                self.detected_obj_name_list.append(classes[int(c)])
        self.detected_obj_name_list = self._handle_duplicates(self.detected_obj_name_list)
        return self.detected_obj_name_list    
    def _get_detected_objs_coord(self):
        detected_objs_coord = {}
        box_center_list = self._get_box_center_coordinate()
        object_num = len(box_center_list)
        for i in range(object_num):
            detected_objs_coord[self.detected_obj_name_list[i]] = box_center_list[i]
        return detected_objs_coord    
    def _get_detected_objs_box(self):
        detected_objs_box_coord = {}
        box_list = self._get_box_coordinates()
        object_num = len(box_list)
        for i in range(object_num):
            detected_objs_box_coord[self.detected_obj_name_list[i]] = box_list[i]
        return detected_objs_box_coord  
    def _get_detected_objs_coord_box(self):
        detected_objs_box_coord = {}
        box_center_list = self._get_box_center_coordinate()
        box_list = self._get_box_coordinates()
        object_num = len(box_list)
        for i in range(object_num):
            detected_objs_box_coord[self.detected_obj_name_list[i]] = {"xyxy":box_list[i],"bb_center":box_center_list[i]}
        return detected_objs_box_coord         
    def _handle_duplicates(self,lst):
        name_count = {}  
        new_lst = []  
        for item in lst: # ['microwave', 'tomato', 'pot']
            if item in name_count:
                name_count[item] += 1  
                new_item = f"{item}_{name_count[item]}"  
            else:
                name_count[item] = 0
                new_item = item
            new_lst.append(new_item)
        return new_lst
    def _get_obj_info_dict(self):
        self._detect()
        self._get_detected_objs()
        self._get_detected_objs_raw()
        detected_objs_box_coord_dict = self._get_detected_objs_coord_box()
        for obj in list(detected_objs_box_coord_dict.keys()):
            xyxy = detected_objs_box_coord_dict[obj]['xyxy']
            bb_img = self.img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:]
            detected_objs_box_coord_dict[obj]['crop_img'] = bb_img
        return detected_objs_box_coord_dict   

def BLIP_vfe(model_vfe,img,device="cuda:0",img_size=224):
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(img).unsqueeze(0).to(device) 
    with torch.no_grad():
        image_feature = model_vfe(image,"",mode='image')[0,0]   

    return image_feature

def BLIP_tfe(model_tfe,text,device="cuda:0"):
    with torch.no_grad():
            text_feature = model_tfe(text,"",mode='text')[0,0]   
    return text_feature         

def obj_info_once(obj_detector, img_feature_exractor, img): # pointclouds/ bbox_center_coordinate are all based on world frame
    object_detection = ObjectDetection(model=obj_detector,img=img, conf=0.7)
    obj_info_dict = object_detection._get_obj_info_dict()
    for obj in list(obj_info_dict.keys()):
        # add visual feature vector of each object into the obj_info_dict
        crop_img_bgr = obj_info_dict[obj]['crop_img']
        crop_img_rgb = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)
        img_feature = BLIP_vfe(img_feature_exractor, crop_img_rgb)
        obj_info_dict[obj]['visual_feature'] = img_feature

    return obj_info_dict    





# class BLIP_vfe(): # RGB image를 받아야함.
#     def __init__(self,model,device="cuda:0",detected_obj_box_image_coord = {},img_size=224):
#         self.device = device
#         self.detected_obj_box_image_coord = detected_obj_box_image_coord
#         self.model_vfe = model
#         self.image_size = img_size
#         self.spatial_relation_dict = {}

#     def _load_image(self,img):
#         img = Image.fromarray(img)
#         # w,h = img.size
#         transform = transforms.Compose([
#             transforms.Resize((self.image_size,self.image_size),interpolation=InterpolationMode.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#             ]) 
#         image = transform(img).unsqueeze(0).to(self.device) 
#         return image         
#     def _VFE(self):
#         for object in list(self.detected_obj_box_image_coord.keys()):
#             cropped_img = self.detected_obj_box_image_coord[object]["crop_img"]
#             image = self._load_image(cropped_img)
#             with torch.no_grad():
#                 image_feature = self.model_vfe(image,"",mode='image')[0,0]
#             self.detected_obj_box_image_coord[object]["image_feature"] = image_feature    
#         return self.detected_obj_box_image_coord      
 
            
# class BLIP_tfe(): # RGB image를 받아야함.
#     def __init__(self,model,text,device="cuda:0"):
#         self.text =text
#         self.device = device
#         self.model_tfe = model     
#     def _TFE(self):

#         with torch.no_grad():
#                 text_feature = self.model_tfe(self.text,"",mode='text')[0,0]

#         return text_feature   


