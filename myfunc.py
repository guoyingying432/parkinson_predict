import numpy as np
import os
import csv


#读数据函数
def read_data(data_path, data_label):
    data_list = os.listdir(data_path)

    N = len(data_list)  # 数据个数
    C = 2  # 坐标点维度(x,y)
    T = 300  # 最大时长（帧）
    V = 21  # 关节点个数

    data_csv = open(data_label)
    data_reader = csv.reader(data_csv)

    total_data = np.zeros((N, C, T, V))
    total_label = []
    count_n = 0

    for (joints_txt, csv_row) in zip(data_list, data_reader):
        #     print("data & label:", joints_txt,csv_row)
        total_label.append(float(csv_row[1]))
        count_t = 0
        for line in open(data_path + joints_txt, "r"):  # 设置文件对象并读取每一行文件
            if count_t >= T:  # 大于最大长度，就退出
                break
            temp_data = line.split()
            try:
                x = np.asarray(temp_data[2::3]).astype(np.float32)[0:21]
                y = np.asarray(temp_data[3::3]).astype(np.float32)[0:21]
            except:
                # 有识别错误的结果则跳过该帧
                x = np.zeros((V))
                y = np.zeros((V))
            if x.shape[0] == V and y.shape[0] == V and float(0) not in x and float(0) not in y:
                total_data[count_n, 0, count_t, :] = x
                total_data[count_n, 1, count_t, :] = y
                count_t += 1
            else:
                pass

        count_n += 1
        # print("complete {}/{}".format(count_n, N))

    total_label = np.asarray(total_label)

    return total_data, total_label

#SG滤波函数
def sgoal_filter(data, fs, window_size, order):
    if window_size == None:
        window_size = fs // 10
    if window_size % 2 == 0 or window_size == 0:
        window_size += 1

    arr = []
    step = int((window_size - 1) / 2)
    for i in range(window_size):
        a = []
        for j in range(order):
            y_val = np.power(-step + i, j)
            a.append(y_val)
        arr.append(a)

    arr = np.mat(arr)
    arr = arr * (arr.T * arr).I * arr.T

    a = np.array(arr[step])
    a = a.reshape(window_size)

    data = np.insert(data, 0, [data[0] for i in range(step)])
    data = np.append(data, [data[-1] for i in range(step)])

    qlist = []
    for i in range(step, data.shape[0] - step):
        arra = []
        for j in range(-step, step + 1):
            arra.append(data[i + j])
        b = np.sum(np.array(arra) * a)
        qlist.append(b)
    return qlist

#convexhull用的函数
def RightTurn(p1, p2, p3):
    if (p3[1] - p1[1]) * (p2[0] - p1[0]) >= (p2[1] - p1[1]) * (p3[0] - p1[0]):
        return False
    return True
def GrahamScan(P):
    P.sort()  # Sort the set of points
    L_upper = [P[0], P[1]]  # Initialize upper part
    # Compute the upper part of the hull
    for i in range(2, len(P)):
        L_upper.append(P[i])
        while len(L_upper) > 2 and not RightTurn(L_upper[-1], L_upper[-2], L_upper[-3]):
            del L_upper[-2]
    L_lower = [P[-1], P[-2]]  # Initialize the lower part
    # Compute the lower part of the hull
    for i in range(len(P) - 3, -1, -1):
        L_lower.append(P[i])
        while len(L_lower) > 2 and not RightTurn(L_lower[-1], L_lower[-2], L_lower[-3]):
            del L_lower[-2]
    del L_lower[0]
    del L_lower[-1]
    L = L_upper + L_lower  # Build the full hull
    return np.array(L)
def GetAreaOfPolyGonbyVector(points):
    # 基于向量叉乘计算多边形面积
    area = 0
    if(len(points)<3):
        raise Exception("error")

    for i in range(0,len(points)-1):
        p1 = points[i]
        p2 = points[i + 1]

        triArea = (p1[0]*p2[1] - p2[0]*p1[1])/2
        #print(triArea)
        area += triArea

    fn=(points[-1][0]*points[0][1]-points[0][0]*points[-1][1])/2
    #print(fn)
    return abs(area+fn)

