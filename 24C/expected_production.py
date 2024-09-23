import pandas as pd
import os
import matplotlib.pyplot as plt

times = 1
XiaomaiYumi_expected_production_growth_rate = 1.075

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.max_rows', None)      # 显示所有行
# pd.set_option('max_colwidth', 400)

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


# 提取需要的字段并建立映射，准备进行模型计算
# 每种作物的亩产量
crop_yield = relevant_data_statistics_2023.set_index(['作物名称', '地块类型', '种植季次'])['亩产量/斤'].to_dict()
# print('crop_yield: \n', crop_yield)
# 每种作物的种植成本
crop_cost = relevant_data_statistics_2023.set_index(['作物名称', '地块类型', '种植季次'])['种植成本/(元/亩)'].to_dict()
# 每种作物的销售价格
crop_price = relevant_data_statistics_2023.set_index(['作物名称', '地块类型', '种植季次'])['销售单价/(元/斤)'].to_dict()

# 在planting_situation_2023中增加一列'地块类型'
planting_situation_2023 = pd.merge(planting_situation_2023, countryside_existing_areas[['种植地块', '地块类型']], on='种植地块', how='left')
# 使用作物名称和地块类型生成“种植价格”列
planting_situation_2023['亩产量/斤'] = planting_situation_2023.apply(lambda row: crop_yield.get((row['作物名称'], row['地块类型'], row['种植季次']), None), axis=1)

# print('planting_situation_2023: \n', planting_situation_2023)

expected_production = {}

for name in countryside_planting_crops['作物名称']:
    df = planting_situation_2023[planting_situation_2023['作物名称'] == name]
    expected_production[name] = sum(df['种植面积/亩']*df['亩产量/斤'])

# print(expected_production)

expected_production['小麦'] = expected_production['小麦'] * (XiaomaiYumi_expected_production_growth_rate ** times)
expected_production['玉米'] = expected_production['玉米'] * (XiaomaiYumi_expected_production_growth_rate ** times)

# print(expected_production)
# print('--------------------------------')
# print(list(expected_production.values())[1])
