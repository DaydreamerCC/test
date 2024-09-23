import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import pulp
import numpy as np
import expected_production

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# 读取上传的Excel文件 附件1和附件2
path = r'E:\2024全国大学生数学建模\C题'
file_1_path = os.path.join(path, '附件1.xlsx')
file_2_path = os.path.join(path, '附件2.xlsx')

countryside_existing_areas = pd.read_excel(file_1_path, sheet_name='乡村的现有耕地')
countryside_planting_crops = pd.read_excel(file_1_path, sheet_name='乡村种植的农作物')
planting_situation_2023 = pd.read_excel(file_2_path, sheet_name='2023年的农作物种植情况')
relevant_data_statistics_2023 = pd.read_excel(file_2_path, sheet_name='2023年统计的相关数据')

# 处理planting_situation_2023表格中的异常数据
for i, x in enumerate(planting_situation_2023['种植地块']):
    if pd.isna(x):
        planting_situation_2023['种植地块'][i] = planting_situation_2023['种植地块'][i-1]
# print('planting_situation_2023[种植地块]: \n', planting_situation_2023['种植地块'])
# print('----------------------------------------------------')

# 在planting_situation_2023中增加一列'地块类型'
planting_situation_2023 = pd.merge(planting_situation_2023, countryside_existing_areas[['种植地块', '地块类型']], on='种植地块', how='left')

# 首先将销售价格区间取平均值，方便计算
# 提取统计数据表中的销售单价信息，并将其从字符串转为数字区间的平均值
relevant_data_statistics_2023['销售单价/(元/斤)'] = relevant_data_statistics_2023['销售单价/(元/斤)'].apply(
    lambda x: sum([float(i) for i in x.split('-')]) / 2 if '-' in str(x) else float(x)
)

# 提取需要的字段并建立映射，准备进行模型计算
# 每种作物的亩产量
crop_yield = relevant_data_statistics_2023.set_index(['作物名称', '地块类型', '种植季次'])['亩产量/斤'].to_dict()
# # 每种作物的种植成本
# crop_cost = relevant_data_statistics_2023.set_index(['作物名称', '地块类型', '种植季次'])['种植成本/(元/亩)'].to_dict()
# # 每种作物的销售价格
# crop_price = relevant_data_statistics_2023.set_index(['作物名称', '地块类型', '种植季次'])['销售单价/(元/斤)'].to_dict()

# 每种作物的种植成本
crop_cost = relevant_data_statistics_2023.set_index(['作物名称'])['种植成本/(元/亩)'].to_dict()
# 每种作物的销售价格
crop_price = relevant_data_statistics_2023.set_index(['作物名称'])['销售单价/(元/斤)'].to_dict()

# 从字典中提取x和y的值
crop_names = list(crop_cost.keys())
costs = list(crop_cost.values())
prices = list(crop_price.values())
ep = list(expected_production.expected_production.values())

# 创建图表
plt.figure(figsize=(10, 6))
plt.scatter(costs, ep, color='blue')

# # 标注每个点的农作物名称
# for i, crop in enumerate(crop_names):
#     plt.text(costs[i], prices[i], crop, fontsize=9, ha='right')

# 设置图表标题和标签
plt.title('Crop Costs vs expected_production')
plt.xlabel('Crop Costs')
plt.ylabel('expected_production')

# 显示图表
plt.show()