import pandas as pd
from math import sin, cos, tan, asin, acos, atan, pi, sqrt
import math
import numpy as np
import random
import csv
from shapely.geometry import Polygon

def Cal_area_2poly(data1, data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """

    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


# D为给定日期从春分算起的天数
D = [306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
# ST为当地时间
ST = [9, 10.5, 12, 13.5, 15]
# z为集热器中心距离任意一个定日镜的竖直距离，为定值
z = 76
# latitude为当地纬度
latitude = math.radians(39.4)
# length为单个镜子的长
length = 6
# width为单个镜子的宽
width = 6

# delta和W(omega)为中间变量，可忽略
delta = [(asin(sin(2*pi*d/365) * sin(2*pi*23.45/360))) for d in D]
W = [pi/12*(st-12) for st in ST]

# 太阳高度角及太阳方位角的计算
Alpha_s = [asin(cos(delta[i]) * cos(latitude) * cos(W[j]) + sin(delta[i]) * sin(latitude))
         for i in range(len(delta)) for j in range(len(W))]     #Alpha_s长度为60
Gamma_s = [acos((sin(delta[i])-sin(Alpha_s[i * len(W) + j]) * sin(latitude)) / (cos(Alpha_s[i * len(W) + j]) * cos(latitude)) + 1e-6)
         for i in range(len(delta)) for j in range(len(W))]     #Gamma_s长度为60

#入射光线的单位方向向量
e_in = [np.array([-cos(Alpha_s[k])*sin(Gamma_s[k]), -cos(Alpha_s[k])*cos(Gamma_s[k]), -sin(Alpha_s[k])]) for k in range(12 * 5)]

excel = pd.read_excel("C:\\Users\\ChenCong\\Desktop\\test.xlsx")
X = list(excel['x'])
Y = list(excel['y'])

#反射光线的单位方向向量
e_out = [np.array([-X[m], -Y[m], z]) / sqrt((X[m])**2 + (Y[m])**2 + z**2) for m in range(len(excel))]

#初始化余弦效率矩阵，大小为60*1745
single_cos_efficiency = np.zeros((60, 1745))
for i in range(60):
    for j in range(1745):
        single_cos_efficiency[i][j] += (1 - (e_in[i] * e_out[j]).sum()) / sqrt(((e_in[i] - e_out[j])*(e_in[i] - e_out[j])).sum())

#num = [random.randint(1, 1743) for i in range(100)]    #随机产生100个镜子A的编号，与镜子A相对应的镜子B编号为镜子A编号+1
num = list(np.arange(0, 1744))

# 镜子A的法向量
e_A = np.zeros((60, len(num), 3))

Alpha_A_m = np.zeros((60, len(num)))
Gamma_A_m = np.zeros((60, len(num)))
Alpha_B_m = np.zeros((60, len(num)))
Gamma_B_m = np.zeros((60, len(num)))
for i in range(60):
    for j in range(len(num)):
        Alpha_A_m[i][j] = acos(((e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum()))[2])
        Gamma_A_m[i][j] = atan(((e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum()))[0] /
                               ((e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum()))[1])
        Alpha_B_m[i][j] = acos(((e_in[i] - e_out[num[j]+1]) / sqrt(((e_in[i] - e_out[num[j]+1])*(e_in[i] - e_out[num[j]+1])).sum()))[2])
        Gamma_B_m[i][j] = atan(((e_in[i] - e_out[num[j]+1]) / sqrt(((e_in[i] - e_out[num[j]+1])*(e_in[i] - e_out[num[j]+1])).sum()))[0] /
                               ((e_in[i] - e_out[num[j]+1]) / sqrt(((e_in[i] - e_out[num[j]+1])*(e_in[i] - e_out[num[j]+1])).sum()))[1])
        e_A[i][j] = (e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum())


# Pbi为镜子B的四个顶点在镜子B坐标系下的坐标
Pb1 = np.array([-length/2, width/2, 0])
Pb2 = np.array([length/2, width/2, 0])
Pb3 = np.array([length/2, -width/2, 0])
Pb4 = np.array([-length/2, -width/2, 0])

# Vbi是大小为60*len(num)的矩阵，存储着镜子B的四个顶点在大坐标系下的坐标
Vb1 = np.zeros((60, len(num), 3))
Vb2 = np.zeros((60, len(num), 3))
Vb3 = np.zeros((60, len(num), 3))
Vb4 = np.zeros((60, len(num), 3))

# Vai为镜子A的四个顶点在大坐标系下的坐标
Va1 = np.array([0, 0, 0])
Va2 = np.array([0, 0, 0])
Va3 = np.array([0, 0, 0])
Va4 = np.array([0, 0, 0])

# 阴影面积存储矩阵
shadow_area = np.zeros((60, len(num)))

for i in range(60):
    for j in range(len(num)):
        # transition_Matrix_B为Pbi --> Vbi的转移矩阵
        transition_Matrix_B = np.array([[cos(Gamma_B_m[i][j]), sin(Gamma_B_m[i][j])*sin(Alpha_B_m[i][j]), -sin(Gamma_B_m[i][j])*cos(Alpha_B_m[i][j])],
                              [-sin(Alpha_B_m[i][j]), cos(Gamma_B_m[i][j])*sin(Alpha_B_m[i][j]), -cos(Gamma_B_m[i][j])*cos(Alpha_B_m[i][j])],
                              [0, cos(Alpha_B_m[i][j]), sin(Alpha_B_m[i][j])]])
        Vb1[i][j] = (np.dot(transition_Matrix_B, Pb1.T) + np.array([X[num[j] + 1], Y[num[j] + 1], 4]).T).T
        Vb2[i][j] = (np.dot(transition_Matrix_B, Pb2.T) + np.array([X[num[j] + 1], Y[num[j] + 1], 4]).T).T
        Vb3[i][j] = (np.dot(transition_Matrix_B, Pb3.T) + np.array([X[num[j] + 1], Y[num[j] + 1], 4]).T).T
        Vb4[i][j] = (np.dot(transition_Matrix_B, Pb4.T) + np.array([X[num[j] + 1], Y[num[j] + 1], 4]).T).T
        #计算B点在A平面上投影时的方程组系数矩阵
        coefficient_matrix = np.array([[e_A[i][j][0], e_A[i][j][1], e_A[i][j][2]],
                                       [cos(Alpha_s[i])*cos(Gamma_s[i]), -cos(Alpha_s[i])*sin(Gamma_s[i]), 0],
                                       [0, sin(Gamma_s[i]), cos(Alpha_s[i])*cos(Gamma_s[i])]])

        #增广矩阵
        augmented_matrix1 = np.array([e_A[i][j][0]*X[num[j]] + e_A[i][j][1]*Y[num[j]] + e_A[i][j][2]*4,
                                      -cos(Alpha_s[i])*sin(Gamma_s[i])*Vb1[i][j][1] + cos(Alpha_s[i])*cos(Gamma_s[i])*Vb1[i][j][0],
                                      sin(Gamma_s[i])*Vb1[i][j][1] - cos(Alpha_s[i])*cos(Gamma_s[i])*Vb1[i][j][2]
                                      ])
        augmented_matrix2 = np.array([e_A[i][j][0]*X[num[j]] + e_A[i][j][1]*Y[num[j]] + e_A[i][j][2]*4,
                                      -cos(Alpha_s[i])*sin(Gamma_s[i])*Vb2[i][j][1] + cos(Alpha_s[i])*cos(Gamma_s[i])*Vb2[i][j][0],
                                      sin(Gamma_s[i])*Vb2[i][j][1] - cos(Alpha_s[i])*cos(Gamma_s[i])*Vb2[i][j][2]
                                      ])
        augmented_matrix3 = np.array([e_A[i][j][0]*X[num[j]] + e_A[i][j][1]*Y[num[j]] + e_A[i][j][2]*4,
                                      -cos(Alpha_s[i])*sin(Gamma_s[i])*Vb3[i][j][1] + cos(Alpha_s[i])*cos(Gamma_s[i])*Vb3[i][j][0],
                                      sin(Gamma_s[i])*Vb3[i][j][1] - cos(Alpha_s[i])*cos(Gamma_s[i])*Vb3[i][j][2]
                                      ])
        augmented_matrix4 = np.array([e_A[i][j][0]*X[num[j]] + e_A[i][j][1]*Y[num[j]] + e_A[i][j][2]*4,
                                      -cos(Alpha_s[i])*sin(Gamma_s[i])*Vb4[i][j][1] + cos(Alpha_s[i])*cos(Gamma_s[i])*Vb4[i][j][0],
                                      sin(Gamma_s[i])*Vb4[i][j][1] - cos(Alpha_s[i])*cos(Gamma_s[i])*Vb4[i][j][2]
                                      ])
        # Tbi为求解出来的投影之后的镜子B的四个顶点的坐标
        Tb1 = np.linalg.solve(coefficient_matrix, augmented_matrix1)
        Tb2 = np.linalg.solve(coefficient_matrix, augmented_matrix2)
        Tb3 = np.linalg.solve(coefficient_matrix, augmented_matrix3)
        Tb4 = np.linalg.solve(coefficient_matrix, augmented_matrix4)

        # transition_Matrix_A为Pai --> Vai的转移矩阵，但由于Pai和Pbi的坐标相同，故这里采用Pbi坐标计算
        transition_Matrix_A = np.array([[cos(Gamma_A_m[i][j]), sin(Gamma_A_m[i][j])*sin(Alpha_A_m[i][j]), -sin(Gamma_A_m[i][j])*cos(Alpha_A_m[i][j])],
                              [-sin(Alpha_A_m[i][j]), cos(Gamma_A_m[i][j])*sin(Alpha_A_m[i][j]), -cos(Gamma_A_m[i][j])*cos(Alpha_A_m[i][j])],
                              [0, cos(Alpha_A_m[i][j]), sin(Alpha_A_m[i][j])]])
        Va1 = (np.dot(transition_Matrix_A, Pb1.T) + np.array([X[num[j]], Y[num[j]], 4]).T).T
        Va2 = (np.dot(transition_Matrix_A, Pb2.T) + np.array([X[num[j]], Y[num[j]], 4]).T).T
        Va3 = (np.dot(transition_Matrix_A, Pb3.T) + np.array([X[num[j]], Y[num[j]], 4]).T).T
        Va4 = (np.dot(transition_Matrix_A, Pb4.T) + np.array([X[num[j]], Y[num[j]], 4]).T).T

        data1 = [Tb1, Tb2, Tb3, Tb4]
        data2 = [Va1, Va2, Va3, Va4]
        # 计算两个凸多边形的重合面积
        shadow_area[i][j] = Cal_area_2poly(data1, data2)

print(shadow_area)
print(shadow_area.mean()*2/36)
# print(shadow_area.shape)
# zero = np.zeros((60, 1))
# new = np.concatenate((shadow_area, zero), 1)
# print(new.shape)


# f = open('table_2.csv', 'a', newline='', encoding='utf-8-sig')
# global writer
# writer = csv.writer(f)
# writer.writerow(['时间', '月平均阴影遮挡效率'])
# for i in range(12):
#     writer.writerow([f'{i+1}月21日', 1 - shadow_area[5*i : 5*i+5].mean()*2/(length*width)])