import pandas as pd
from math import sin, cos, tan, asin, acos, atan, pi, sqrt
import math
import numpy as np
import random
import csv
from tqdm import tqdm
import time


# D为给定日期从春分算起的天数
D = [306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
# ST为当地时间
ST = [9, 10.5, 12, 13.5, 15]
# z为集热器中心距离任意一个定日镜中心的竖直距离，为定值
z = 76
# latitude为当地纬度
latitude = math.radians(39.4)
# length为单个镜子的长
length = 6
# width为单个镜子的宽
width = 6
# R为集热器半径
R = 3.5
# Ht为集热器中心高度
Ht = 80
# Lt为集热器长度
Lt = 8

# delta和W(omega)为中间变量，可忽略
delta = [(asin(sin(2*pi*d/365) * sin(2*pi*23.45/360))) for d in D]
W = [pi/12*(st-12) for st in ST]

# 太阳高度角及太阳方位角的计算
Alpha_s = [asin(cos(delta[i]) * cos(latitude) * cos(W[j]) + sin(delta[i]) * sin(latitude))
         for i in range(len(delta)) for j in range(len(W))]     # Alpha_s长度为60
Gamma_s = [acos((sin(delta[i])-sin(Alpha_s[i * len(W) + j]) * sin(latitude)) / (cos(Alpha_s[i * len(W) + j]) * cos(latitude)) + 1e-6)
         for i in range(len(delta)) for j in range(len(W))]     # Gamma_s长度为60

#入射光线的单位方向向量
e_in = [np.array([-cos(Alpha_s[k])*sin(Gamma_s[k]), -cos(Alpha_s[k])*cos(Gamma_s[k]), -sin(Alpha_s[k])]) for k in range(12 * 5)]

excel = pd.read_excel("C:\\Users\\ChenCong\\Desktop\\test.xlsx")
X = list(excel['x'])
Y = list(excel['y'])

#反射光线的单位方向向量，长度为1745
e_out = [np.array([-X[m], -Y[m], z]) / sqrt((X[m])**2 + (Y[m])**2 + z**2) for m in range(len(excel))]

num = list(np.arange(0, 1, 1))
print(num)

# 镜子A的法向量
e_A = np.zeros((60, len(num), 3))

Alpha_A_m = np.zeros((60, len(num)))
Gamma_A_m = np.zeros((60, len(num)))

for i in range(60):
    for j in range(len(num)):
        Alpha_A_m[i][j] = acos(((e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum()))[2])
        Gamma_A_m[i][j] = atan(((e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum()))[0] /
                               ((e_in[i] - e_out[num[j]]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum()))[1])
        e_A[i][j] = (e_out[num[j]] - e_in[i]) / sqrt(((e_in[i] - e_out[num[j]])*(e_in[i] - e_out[num[j]])).sum())

#print(e_A)

# 截断效率
truncation_efficiency = np.zeros((60, len(num)))

for i in tqdm(range(60)):
    for j in range(len(num)):
        light_hit_sum = 0
        for k in range(100):
            x_A = random.uniform(-3, 3)
            y_A = random.uniform(-3, 3)
            z_A = 0
            P = np.array([x_A, y_A, z_A])
            #print(P)
            alpha_t = random.uniform(0, 0.00465)
            gamma_t = random.uniform(0, 2*pi)
            e_in_shift = np.array([-sin(alpha_t)*sin(gamma_t), -sin(alpha_t)*cos(gamma_t), -cos(alpha_t)]).T

            # transition_Matrix_s = np.array([[cos(Gamma_s[i]), sin(Gamma_s[i])*sin(Alpha_s[i]), -sin(Gamma_s[i])*cos(Alpha_s[i])],
            #                       [-sin(Alpha_s[i]), cos(Gamma_s[i])*sin(Alpha_s[i]), -cos(Gamma_s[i])*cos(Alpha_s[i])],
            #                       [0, cos(Alpha_s[i]), sin(Alpha_s[i])]])

            transition_Matrix_s = np.array([[cos(Gamma_s[i]), sin(Gamma_s[i])*cos(Alpha_s[i]), sin(Gamma_s[i])*sin(Alpha_s[i])],
                                  [-sin(Gamma_s[i]), cos(Gamma_s[i])*cos(Alpha_s[i]), cos(Gamma_s[i])*sin(Alpha_s[i])],
                                  [0, -sin(Alpha_s[i]), cos(Alpha_s[i])]])

            e_in_shift_transition = np.dot(transition_Matrix_s, e_in_shift)
            #print((e_in_shift_transition*e_in_shift_transition).sum())
            #print(e_in_shift_transition)

            # transition_Matrix_a = np.array([[cos(Gamma_A_m[i][j]), sin(Gamma_A_m[i][j])*sin(Alpha_A_m[i][j]), -sin(Gamma_A_m[i][j])*cos(Alpha_A_m[i][j])],
            #                       [-sin(Alpha_A_m[i][j]), cos(Gamma_A_m[i][j])*sin(Alpha_A_m[i][j]), -cos(Gamma_A_m[i][j])*cos(Alpha_A_m[i][j])],
            #                       [0, cos(Alpha_A_m[i][j]), sin(Alpha_A_m[i][j])]])

            transition_Matrix_a = np.array([[cos(Gamma_A_m[i][j]), sin(Gamma_A_m[i][j])*cos(Alpha_A_m[i][j]), sin(Gamma_A_m[i][j])*sin(Alpha_A_m[i][j])],
                                  [-sin(Gamma_A_m[i][j]), cos(Gamma_A_m[i][j])*cos(Alpha_A_m[i][j]), cos(Gamma_A_m[i][j])*sin(Alpha_A_m[i][j])],
                                  [0, -sin(Alpha_A_m[i][j]), cos(Alpha_A_m[i][j])]])

            P_transition = np.dot(transition_Matrix_a, P.T) + np.array([X[num[j]], Y[num[j]], 4]).T
            print(P_transition)

            e_out_shift_transition = e_in_shift_transition - (2 * ((e_in_shift_transition * e_A[i][j]).sum()) * e_A[i][j])

            #print(e_out_shift_transition)

            #print((e_out_shift_transition*e_out_shift_transition).sum())

            # triangle_delta = (4*(e_out_shift_transition[0]*P_transition[0] + e_out_shift_transition[1]*P_transition[1])**2
            #                   - 4*(e_out_shift_transition[0]+e_out_shift_transition[1])*(P_transition[0]**2+P_transition[1]**2-R**2))

            triangle_delta = (4*(e_out_shift_transition[0]**2+e_out_shift_transition[1]**2)*(R**2) - 4*(e_out_shift_transition[0]*P_transition[1]-P_transition[0]*e_out_shift_transition[1])**2) / (e_out_shift_transition[1]**2)

            min = (Ht - 0.5*Lt - P_transition[2])/e_out_shift_transition[2]
            max = (Ht + 0.5*Lt - P_transition[2])/e_out_shift_transition[2]

            # print(f'min = {min}, max = {max}')
            # print(f'triangle_delta = {triangle_delta}')
            # if triangle_delta >= 0:
            #     temp1 = (-(e_out_shift_transition[0] * P_transition[0] + e_out_shift_transition[1] * P_transition[1]) + sqrt(triangle_delta)) / (e_out_shift_transition[0] ** 2 + e_out_shift_transition[1] ** 2)
            #     temp2 = (-(e_out_shift_transition[0] * P_transition[0] + e_out_shift_transition[1] * P_transition[1]) - sqrt(triangle_delta)) / (e_out_shift_transition[0] ** 2 + e_out_shift_transition[1] ** 2)
            #     print(f'temp1 = {temp1}, temp2 = {temp2}')

            if triangle_delta >= 0:
                temp1 = (-2*(e_out_shift_transition[0] * P_transition[0] + e_out_shift_transition[1] * P_transition[1]) + e_out_shift_transition[1]**2 * sqrt(triangle_delta)) / (2*(e_out_shift_transition[0]**2 + e_out_shift_transition[1]**2))
                temp2 = (-2*(e_out_shift_transition[0] * P_transition[0] + e_out_shift_transition[1] * P_transition[1]) - e_out_shift_transition[1]**2 * sqrt(triangle_delta)) / (2*(e_out_shift_transition[0]**2 + e_out_shift_transition[1]**2))
                if (temp1 >= min and temp1 <= max) or (temp2 >= min and temp2 <= max):
                    light_hit_sum = light_hit_sum + 1
        truncation_efficiency[i][j] = light_hit_sum / 100

# print(truncation_efficiency.mean())
# print(truncation_efficiency)