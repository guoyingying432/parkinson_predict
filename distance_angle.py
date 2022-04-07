import numpy as np
import math
import myfunc
import pandas as pd

#读取数据
data_path1 = "data1/joints/"  # 骨架数据的位置
data_label1 = "data1/data1_label.csv"  # label文件的位置，label文件中的顺序和文件名一一对应
data_path2 = "data2/joints/"  # 骨架数据的位置
data_label2 = "data2/data2_label.csv"  # label文件的位置，label文件中的顺序和文件名一一对应
total_data1, total_label1 = myfunc.read_data(data_path1, data_label1)
total_data2, total_label2 = myfunc.read_data(data_path2, data_label2)

total_data = np.concatenate((total_data1,total_data2),axis=0)
total_label = np.concatenate((total_label1,total_label2),axis=0)
N = total_data.shape[0]

dis = np.zeros((N,300))
dis_std = np.zeros(N)
dis_mean = np.zeros(N)
dis_max = np.zeros(N)
dis_min = np.zeros(N)
#食指指尖和大拇指距离
for k in range(0,N):
    dis_list=[]
    length = (total_data[k, 0, :, 4] == 0).argmax(axis=0)  # 实际帧数
    if length == 0:
        length = 300
    for i in range(0, length-1):
        x1 = total_data[k, :, i, :][0][4]
        y1 = -total_data[k, :, i, :][1][4]
        x2 = total_data[k, :, i+1, :][0][8]
        y2 = -total_data[k, :, i+1, :][1][8]
        distance = math.sqrt(abs(int(x1)-int(x2))**2+abs(int(y1)-int(y2))**2)
        dis[k, i] = distance
        dis_list.append(distance)

    dis_std[k] = np.std(dis_list)
    dis_max[k] = np.max(dis_list)
    dis_min[k] = np.min(dis_list)
    dis_mean[k] = np.mean(dis_list)

df_dis = pd.concat([pd.DataFrame(dis_mean,columns=['dis_mean']),
               pd.DataFrame(dis_max,columns=['dis_max']),
               pd.DataFrame(dis_min,columns=['dis_min']),
               pd.DataFrame(dis_std,columns=['dis_std']),
               pd.DataFrame(total_label,columns=['total_label'])],axis=1)

print(df_dis.corr('spearman'))  #spearman


#开合角度列表
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

angle1 = np.zeros((N,300))
angle1_std = np.zeros(N)
angle1_mean = np.zeros(N)
angle1_max = np.zeros(N)
angle1_min = np.zeros(N)

for k in range(0,N):
    ang_list=[]
    length = (total_data[k, 0, :, 4] == 0).argmax(axis=0)  # 实际帧数
    if length == 0:
        length = 300
    for i in range(0, length-1):
        x1 = total_data[0, :, i, :][0][4]
        y1 = -total_data[0, :, i, :][1][4]
        x2 = total_data[0, :, i+1, :][0][8]
        y2 = -total_data[0, :, i+1, :][1][8]
        x3 = total_data[0, :, i, :][0][0]
        y3 = -total_data[0, :, i, :][1][0]
        muzhi=[x3,y3,x1,y1]
        shizhi=[x3,y3,x2,y2]
        ang=angle(muzhi,shizhi)
        angle1[k, i] = ang
        ang_list.append(ang)

    angle1_std[k] = np.std(ang_list)
    angle1_max[k] = np.max(ang_list)
    angle1_min[k] = np.min(ang_list)
    angle1_mean[k] = np.mean(ang_list)

df_angle = pd.concat([pd.DataFrame(angle1_mean,columns=['angle_mean']),
               pd.DataFrame(angle1_max,columns=['angle_max']),
               pd.DataFrame(angle1_min,columns=['angle_min']),
               pd.DataFrame(angle1_std,columns=['angle_std']),
               pd.DataFrame(total_label,columns=['total_label']),],axis=1)

print(df_angle.corr('spearman'))