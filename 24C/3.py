import pandas as pd
import os
import matplotlib.pyplot as plt
import pulp
import expected_production
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

which_season = {1: '第一季', 2: '第二季'}

times = 1

XiaomaiYumi_expected_production_growth_rate = 1.075
planting_cost_growth_rate = 1.05
vegetable_price_growth_rate = 1.05
fungi_price = 0.95

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
# print('crop_cost: ', crop_cost)
# print('crop_price: ', crop_price)
# print('expected_production.expected_production: ', expected_production.expected_production)
# print([list(crop_price.values())[i] * list(expected_production.expected_production.values())[i] for i in range(41)])


# 假设 a, b, c 是你的数据
a = np.array(list(crop_cost.values()))
b = np.array(list(crop_price.values()))
c = np.array(list(expected_production.expected_production.values()))

# 将 a 和 b 组合成一个二维数组作为自变量
X = np.column_stack((a, b))

# 创建回归模型
model = LinearRegression()
model.fit(X, c)

# 输出回归系数
print("回归系数:", model.coef_)
print("截距:", model.intercept_)

# 预测值
c_pred = model.predict(X)

# 计算 R^2 评分 (判定系数)
r2 = r2_score(c, c_pred)
print("R^2:", r2)


crop_price = {'黄豆': 3.25, '黑豆': 7.5, '红豆': 8.25, '绿豆': 7.0, '爬豆': 6.75, '小麦': 3.5, '玉米': 3.0, '谷子': 6.75,
              '高粱': 6.0, '黍子': 7.5, '荞麦': 40.0, '南瓜': 1.5, '红薯': 3.25, '莜麦': 5.5, '大麦': 3.5, '水稻': 7.0,
              '豇豆': 8.0, '刀豆': 6.75, '芸豆': 6.5, '土豆': 3.75, '西红柿': 6.25, '茄子': 5.5, '菠菜 ': 5.75,
              '青椒': 5.25, '菜花': 5.5, '包菜': 6.5, '油麦菜': 5.0, '小青菜': 5.75, '黄瓜': 7.0, '生菜': 5.25,
              '辣椒': 7.25, '空心菜': 4.5, '黄心菜': 4.5, '芹菜': 4.0, '大白菜': 2.5, '白萝卜': 2.5, '红萝卜': 3.25,
              '榆黄菇': 57.5, '香菇': 19.0, '白灵菇': 16.0, '羊肚菌': 100.0}

crop_cost = {'黄豆': 400, '黑豆': 400, '红豆': 350, '绿豆': 350, '爬豆': 350, '小麦': 450, '玉米': 500, '谷子': 360,
             '高粱': 400, '黍子': 360, '荞麦': 350, '南瓜': 1000, '红薯': 2000, '莜麦': 400, '大麦': 350, '水稻': 680,
             '豇豆': 2400, '刀豆': 1200, '芸豆': 2400, '土豆': 2400, '西红柿': 2400, '茄子': 2400, '菠菜 ': 2700,
             '青椒': 2000, '菜花': 3000, '包菜': 3500, '油麦菜': 2000, '小青菜': 2000, '黄瓜': 3500, '生菜': 2000,
             '辣椒': 1200, '空心菜': 5000, '黄心菜': 2500, '芹菜': 1100, '大白菜': 2000, '白萝卜': 500, '红萝卜': 500,
             '榆黄菇': 3000, '香菇': 2000, '白灵菇': 10000, '羊肚菌': 10000}

expected_production = {'黄豆': 57000.0, '黑豆': 21850.0, '红豆': 22400.0, '绿豆': 33040.0, '爬豆': 9875.0, '小麦': 183653.0,
                       '玉米': 142706.25, '谷子': 71400.0, '高粱': 30000.0, '黍子': 12500.0, '荞麦': 1500.0, '南瓜': 35100.0,
                       '红薯': 36000.0, '莜麦': 14000.0, '大麦': 10000.0, '水稻': 21000.0, '豇豆': 36480.0, '刀豆': 26880.0,
                       '芸豆': 6480.0, '土豆': 30000.0, '西红柿': 36210.0, '茄子': 45360.0, '菠菜 ': 900.0, '青椒': 2610.0,
                       '菜花': 3600.0, '包菜': 4050.0, '油麦菜': 4500.0, '小青菜': 35480.0, '黄瓜': 13050.0, '生菜': 0,
                       '辣椒': 1200.0, '空心菜': 3600.0, '黄心菜': 1800.0, '芹菜': 1800.0, '大白菜': 150000.0, '白萝卜': 100000.0,
                       '红萝卜': 36000.0, '榆黄菇': 9000.0, '香菇': 7200.0, '白灵菇': 18000.0, '羊肚菌': 4200.0}

# 创建 DataFrame 存储数据
df = pd.DataFrame({
    'Price': pd.Series(crop_price),
    'Cost': pd.Series(crop_cost),
    'Expected_Production': pd.Series(expected_production)
})

# 计算秩次
df['Price_Rank'] = df['Price'].rank()
df['Cost_Rank'] = df['Cost'].rank()
df['Production_Rank'] = df['Expected_Production'].rank()

# 计算价格和成本、价格和产量的秩次差
df['Price_Cost_Diff'] = df['Price_Rank'] - df['Cost_Rank']
df['Price_Production_Diff'] = df['Price_Rank'] - df['Production_Rank']

# 计算 d^2
df['Price_Cost_Diff_Squared'] = df['Price_Cost_Diff'] ** 2
df['Price_Production_Diff_Squared'] = df['Price_Production_Diff'] ** 2

# 计算斯皮尔曼秩相关系数
n = len(df)
spearman_price_cost = 1 - (6 * df['Price_Cost_Diff_Squared'].sum()) / (n * (n**2 - 1))
spearman_price_production = 1 - (6 * df['Price_Production_Diff_Squared'].sum()) / (n * (n**2 - 1))

print('spearman_price_cost: ', spearman_price_cost)
print('spearman_price_production: ', spearman_price_production)

# 种植成本每年增加
for key in crop_cost:
    crop_cost[key] *= (planting_cost_growth_rate ** times)

print(crop_price)
for key in crop_price:
    crop_price[key] *= (vegetable_price_growth_rate ** times)
print(crop_price)


# 每一块田的面积
every_area = countryside_existing_areas['地块面积/亩']
# print(every_area[0])


# 以下代码中，下标i表示第i种作物，下标j表示第j块地，下标k表示第k季，下标t表示第t年
# 同时，为了与建模的下标对应，所有下标均从1开始，因此所有的range均形如range(1, x)，所有的list[]均形如list[x-1]
# 对于作物下标i（1~41）: 1~5为粮食（豆类），6~15为粮食类，16为水稻（粮食）
#                    17~37为蔬菜（豆类），20~34为蔬菜，35~37为萝卜（蔬菜）
#                    38~41为食用菌
# 对于地块下标j（1~54）: 1~6为平旱地，7~20为梯田，21~26为山坡地
#                    27~34为水浇地
#                    35~50为普通大棚，51~54为智慧大棚
# 对于下标k（1，2）: 1表示第一季，2表示第二季（如果作物为单季作物则有x[k==2]==0）
# 对于下标t（1~7）: t=1对应着2024年，t=7对应着2030年


model = pulp.LpProblem("Maximize_Total_income", pulp.LpMaximize)

x = pulp.LpVariable.dicts("x", [(i, j, k, t) for i in range(1, 42) for j in range(1, 55) for k in range(1, 3) for t in range(1, 8)],
                             lowBound=0, cat='Continuous')

# 决策变量b1[(i, j, t)]，决定干田上是否种植豆类（如果种植，该块干田全部种植豆类）
b1 = pulp.LpVariable.dicts("b1", [(i, j, t) for i in range(1, 6) for j in range(1, 27) for t in range(1, 8)],
                             lowBound=0, cat='Binary')

# 决策变量b2[(i, j, t)]，决定第j块干田第t年是否种植第i种作物（任意一块干田不能连续两年种植同一种作物）
b2 = pulp.LpVariable.dicts("b2", [(i, j, t) for i in range(1, 16) for j in range(1, 27) for t in range(1, 7)],
                             lowBound=0, cat='Binary')

# 决策变量b3[(i, j, t)]，决定第j块水浇地第t年是否种植水稻（水稻只能种植一季）
b3 = pulp.LpVariable.dicts("b3", [(i, j, t) for i in range(16, 17) for j in range(27, 35) for t in range(1, 7)],
                             lowBound=0, cat='Binary')

# 决策变量b4[(i, j, t)]，决定水浇地上是否种植豆类（任意一块水浇地最近三年之内要种植过豆类）
b4 = pulp.LpVariable.dicts("b4", [(i, j, t) for i in range(17, 20) for j in range(27, 35) for t in range(1, 8)],
                             lowBound=0, cat='Binary')

# 决策变量b5[(i, j, t)]，决定同年的两季智慧大棚是否种植第i种作物(智能大棚两季不能种植同一种作物)
b5 = pulp.LpVariable.dicts("b5", [(i, j, t) for i in range(17, 35) for j in range(51, 55) for t in range(1, 8)],
                             lowBound=0, cat='Binary')

# 决策变量b6[(i, j, t)]，决定隔年的两季智慧大棚是否种植第i种作物(智能大棚两季不能种植同一种作物)
b6 = pulp.LpVariable.dicts("b6", [(i, j, t) for i in range(17, 35) for j in range(51, 55) for t in range(1, 7)],
                             lowBound=0, cat='Binary')

# 决策变量b7[(i, j, t)]，决定智慧大棚是否种植豆类（任意一块水浇地最近三年之内要种植过豆类）
b7 = pulp.LpVariable.dicts("b7", [(i, j, k, t) for i in range(17, 20) for j in range(51, 55) for k in range(1, 3) for t in range(1, 8)],
                             lowBound=0, cat='Binary')




# 任意一块干田上作物的面积应小于该块干田的面积
for j in range(1, 27):
    for k in range(1, 2):
        for t in range(1, 8):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(1, 16)) <= every_area[j - 1]

# 任意一块干田最近三年之内要种植过豆类
for j in range(1, 27):
    for k in range(1, 2):
        for t in range(1, 6):
            model += pulp.lpSum(x[(i, j, k, t)] for t in range(t, t+3) for i in range(1, 6)) >= every_area[j - 1]

# 任意一块干田不能连续两年种植同一种作物
for i in range(1, 16):
    for j in range(1, 27):
        for k in range(1, 2):
            for t in range(1, 7):
                model += x[(i, j, k, t)] <= every_area[j - 1] * b2[(i, j, t)]
                model += x[(i, j, k, t+1)] <= every_area[j - 1] * (1 - b2[(i, j, t)])

# 水稻只能种植一季
for i in range(16, 17):
    for j in range(27, 35):
        for k in range(1, 2):
            for t in range(1, 7):
                model += x[(i, j, k, t)] <= every_area[j - 1] * b3[(i, j, t)]
                model += x[(i, j, k, t + 1)] <= every_area[j - 1] * (1 - b3[(i, j, t)])

# 任意一块水浇地最近三年之内要种植过豆类
for j in range(27, 35):
    for k in range(1, 2):
        for t in range(1, 6):
            model += pulp.lpSum(x[(i, j, k, t)] for t in range(t, t+3) for i in range(17, 20)) >= every_area[j - 1]

# 任意一块水浇地上作物的面积应小于该块水浇地的面积（第一季）
for j in range(27, 35):
    for k in range(1, 2):
        for t in range(1, 8):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(17, 35)) <= every_area[j - 1]

# 任意一块水浇地上作物的面积应小于该块水浇地的面积（第二季）（水浇地两季作物不同）
for j in range(27, 35):
    for k in range(2, 3):
        for t in range(1, 8):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(35, 38)) <= every_area[j - 1]

# 任意一块普通大棚作物的面积应小于该块普通大棚的面积（第一季）
for j in range(17, 35):
    for k in range(1, 2):
        for t in range(1, 8):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(17, 35)) <= every_area[j - 1]

# 任意一块普通大棚作物的面积应小于该块普通大棚的面积（第二季）（普通大棚两季作物不同）
for j in range(35, 51):
    for k in range(2, 3):
        for t in range(1, 8):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(38, 42)) <= every_area[j - 1]

# 任意一块普通大棚最近三年之内要种植过豆类
for j in range(35, 51):
    for t in range(1, 6):
        model += pulp.lpSum(x[(i, j, 1, t)] for t in range(t, t+3) for i in range(17, 20)) >= every_area[j - 1]

# 任意一块智能大棚作物的面积应小于该块智能大棚的面积（两季）
for j in range(51, 55):
    for k in range(1, 3):
        for t in range(1, 7):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(17, 35)) <= every_area[j - 1]

# 智能大棚两季不能种植同一种作物
for i in range(17, 35):
    for j in range(51, 55):
        for t in range(1, 7):
            # 智能大棚隔年的两季不能种植同一种作物
            model += x[(i, j, 2, t)] <= every_area[j - 1] * b6[(i, j, t)]
            model += x[(i, j, 1, t+1)] <= every_area[j - 1] * (1 - b6[(i, j, t)])

# 同上
for i in range(17, 35):
    for j in range(51, 55):
        for t in range(1, 8):
            # 智能大棚同年的两季不能种植同一种作物
            model += x[(i, j, 1, t)] <= every_area[j - 1] * b5[(i, j, t)]
            model += x[(i, j, 2, t)] <= every_area[j - 1] * (1 - b5[(i, j, t)])

# 任意一块智慧大棚最近三年之内要种植过豆类
for j in range(51, 55):
    for t in range(1, 6):
        model += pulp.lpSum(x[(i, j, 1, t)] for t in range(t, t+3) for i in range(17, 20)) + pulp.lpSum(x[(i, j, 2, t)] for t in range(t, t+3) for i in range(17, 20)) >= every_area[j - 1]


"""注意！以下几行代码仅用于第一问第一种情况！"""
# 每年各种农作物产量应小于等于预期销量
for i in range(1, 42):
    for t in range(1, 8):
        model += pulp.lpSum(x[(i, j, k, t)] * crop_yield.get((countryside_planting_crops['作物名称'][i-1], countryside_existing_areas['地块类型'][j-1], which_season[k]), 0) for j in range(1, 55) for k in range(1, 3)) <= list(expected_production.values())[i - 1]


"""以下约束made by Shoucheng Zhu"""
# 粮食类农作物只能种一季
for i in range(1, 16):
    for j in range(1, 27):
        for k in range(2, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 粮食类农作物不能种在大棚和水浇田里
for i in range(1, 16):
    for j in range(27, 55):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 水稻不能种在干田里
for i in range(16, 17):
    for j in range(1, 27):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 水稻不能种在大棚里
for i in range(16, 17):
    for j in range(35, 55):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 水稻只能种一季
for i in range(16, 17):
    for j in range(1, 55):
        for k in range(2, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 蔬菜（除萝卜，白菜，红萝卜）不能种在干田里
for i in range(17, 35):
    for j in range(1, 27):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 蔬菜（除萝卜，白菜，红萝卜）不能种普通大棚和水浇地的第二季
for i in range(17, 35):
    for j in range(27, 51):
        for k in range(2, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 萝卜，白菜，红萝不能种在干田里
for i in range(35, 38):
    for j in range(1, 27):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 萝卜，白菜，红萝卜不能种在水浇地的第一季
for i in range(35, 38):
    for j in range(27, 35):
        for k in range(1, 2):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 萝卜，白菜，红萝卜不能种在大棚里
for i in range(35, 38):
    for j in range(35, 55):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 菌类不能种在干田和水浇地里
for i in range(38, 42):
    for j in range(1, 35):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 菌类不能种在智能大棚里
for i in range(38, 42):
    for j in range(51, 54):
        for k in range(1, 3):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0

# 菌类不能种在普通大棚的第一季
for i in range(38, 42):
    for j in range(35, 51):
        for k in range(1, 2):
            for t in range(1, 8):
                model += x[(i, j, k, t)] == 0



for j in range(1, 55):
    for k in range(1, 3):
        for t in range(1, 8):
            model += pulp.lpSum(x[(i, j, k, t)] for i in range(1, 42)) <= every_area[j - 1]





# crop_yield: 单位亩产量      crop_price: 销售单价      crop_cost: 种植成本
# countryside_planting_crops['作物名称']: 41种作物的名称
# countryside_existing_areas['地块类型']: 54块地的类型
# which_season: 某一季
income = pulp.lpSum(x[(i, j, k, t)] * crop_yield.get((countryside_planting_crops['作物名称'][i-1], countryside_existing_areas['地块类型'][j-1], which_season[k]), 0) * crop_price.get((countryside_planting_crops['作物名称'][i-1], countryside_existing_areas['地块类型'][j-1], which_season[k]), 0)
                    for i in range(1, 42) for j in range(1, 55) for k in range(1, 3) for t in range(1, 8))
outcome = pulp.lpSum(x[(i, j, k, t)] * crop_cost.get((countryside_planting_crops['作物名称'][i-1], countryside_existing_areas['地块类型'][j-1], which_season[k]), 0)
                    for i in range(1, 42) for j in range(1, 55) for k in range(1, 3) for t in range(1, 8))

model += income - outcome


# 求解模型
model.solve()

########################################################################################################################
# 输出结果
print("Status:", pulp.LpStatus[model.status])
print("Total Cost:", pulp.value(model.objective))