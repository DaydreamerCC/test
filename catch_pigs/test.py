import pandas as pd
from math import sin, cos, tan, asin, acos, atan, pi, sqrt
import math
import numpy as np
import csv

#D为给定日期从春分算起的天数
D = [306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
#ST为当地时间
ST = [9, 10.5, 12, 13.5, 15]
#z为集热器中心距离任意一个定日镜的竖直距离，为定值
z = 76
#latitude为当地纬度
latitude = math.radians(39.4)

delta = [(asin(sin(2*pi*d/365) * sin(2*pi*23.45/360))) for d in D]

W = [pi/12*(st-12) for st in ST]

Alpha = [asin(cos(delta[i]) * cos(latitude) * cos(W[j]) + sin(delta[i]) * sin(latitude))
         for i in range(len(delta)) for j in range(len(W))]
Gamma = [acos((sin(delta[i])-sin(Alpha[i * len(W) + j]) * sin(latitude)) / (cos(Alpha[i * len(W) + j]) * cos(latitude)) + 1e-6)
         for i in range(len(delta)) for j in range(len(W))]

#入射光线的单位方向向量
e_in = [np.array([-cos(Alpha[k])*sin(Gamma[k]), -cos(Alpha[k])*cos(Gamma[k]), -sin(Alpha[k])]) for k in range(12 * 5)]

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
print(single_cos_efficiency)

monthly_average_cos_efficiency = [single_cos_efficiency[5*i : 5*i+5].mean() for i in range(12)]
print('十二个月的月平均效率为：', monthly_average_cos_efficiency)

f = open('table_1.csv', 'a', newline='', encoding='utf-8-sig')
global writer
writer = csv.writer(f)
writer.writerow(['时间', '月平均效率'])
for i in range(12):
    writer.writerow([f'{i+1}月21日', monthly_average_cos_efficiency[i]])