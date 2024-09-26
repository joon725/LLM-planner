# camera

import numpy as np

def convert_depth_to_pointclouds(depth_frame, u1, u2, v1, v2, fx, fy, cx, cy): # input : (w,h) shape 의 넘파이 depth frame & depth frame에서 목표로 할 범위((u1,v1) 부터 (u2,v2) : bbox의 xyxy에 해당) -> output : camera frame 기준 (x,y,z) 점들로 채워진 리스트 반환
    points = []
    u_range = list(range(u1,u2+1))
    v_range = list(range(v1,v2+1))
    for v in v_range:
        for u in u_range:
            z = depth_frame[v,u]
            if z == 0 :
                continue
            point = convert_depth_to_camera_frame(u,v,z,fx,fy,cx,cy)
            points.append(point) 
    return points        

def convert_depth_to_camera_frame(u,v,z,fx,fy,cx,cy):
    X = (u-cx)*z / fx
    Y = (v-cy)*z / fy
    point = (X,Y,z)
    return point

def transform_pointclouds(points, HT): # input : 특정 frame 기준 (x,y,z) 점들로 채워진 리스트 -> output : 이 점들을 HT 행렬을 이용해 변환한 (x,y,z) 점들의 리스트
    transformed_points = []
    for point in points:
        transformed_point = transform_point(point,HT)
        transformed_points.append(transformed_point)    
    return transformed_points    

def transform_point(point,HT): # 특정 (x,y,z) point를 특정 HT 행렬에 따라 변환
    P_np = np.array([point[0], point[1], point[2], 1])
    P_transformed = np.dot(HT, P_np)
    transformed_point = tuple(P_transformed[:3])
    return transformed_point

    



    

