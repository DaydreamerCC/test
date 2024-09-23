# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# @Time: 2024/8/9 15:28
import os
import pandas as pd
import pulp
# 变量命名
# CDC总仓: central warehouse, 简写成cw
# RDC分仓: regional warehouse, 简写成rw
# 门店: store, 简写成s

# 定义题目数据的原始路径，对于不同用户需要调整
input_data_path = r"C:\Users\ChenCong\Desktop\2024年度“火花杯”数学建模精英联赛-C题-附件\2024年度“火花杯”数学建模精英联赛-C题-附件\input_data"
output_result_path = r"C:\Users\ChenCong\Desktop\2024年度“火花杯”数学建模精英联赛-C题-附件\2024年度“火花杯”数学建模精英联赛-C题-附件\output_result"

# 定义单位运输成本(单位：元/(吨*公里))
unit_cost_cw_rw, unit_cost_cw_s, unit_cost_rw_s = 0.6, 1.25, 1.25
valid_oder_time = 10 * 60  # 订单需求时效(单位：分钟)
time_satisfaction_rate = 0.95  # 时效满足率95%
single_smart_system_cost = 10000  # 智能调度系统成本
# 时效性惩罚，自定义默认值1000000，取个较大的常数即可
penalty_val = 2000000
# 时间满足度约束
M = 10000  # 大M

# 读取df_wh，获取仓库和门店列表
path = os.path.join(input_data_path, 'df_wh.csv')
df_wh = pd.read_csv(path, encoding='gbk')
central_warehouses = df_wh.loc[df_wh['Type'] == 'CDC']['Name'].values.tolist()
regional_warehouses = df_wh.loc[df_wh['Type'] == 'RDC']['Name'].values.tolist()
df_customer = pd.read_csv(os.path.join(input_data_path, 'df_customer.csv'))
stores = df_customer['Name'].values.tolist()

# print(central_warehouses)
# print(regional_warehouses)
# print(stores)

# 读取df_matrix, 计算仓库和门店之间的单位运输成本(单位：元/吨)
df_matrix = pd.read_csv(os.path.join(input_data_path, 'df_matrix.csv'))
df_matrix['unit_cost'] = df_matrix.apply(
    lambda x: x['Distance'] * unit_cost_cw_rw if x['To'] in regional_warehouses else x['Distance'] * unit_cost_rw_s,
    axis=1)
#print(df_matrix)

# 计算仓库到门店的时效满足性，1表示满足，0表示不满足
df_matrix['valid_duration'] = df_matrix.apply(lambda x: 1 if x['Duration'] <= valid_oder_time else 0, axis=1)
#print(df_matrix)

# ----------------------------------------------------------------------------------------------------------------------
# 用字典保存仓库到门店的单位运输成本(单位：元/吨)
transport_costs = {name: group.set_index('To')['unit_cost'].to_dict() for name, group in df_matrix.groupby('From')}
# 时效约束
time_constraints = {name: group.set_index('To')['valid_duration'].to_dict() for name, group in df_matrix.groupby('From')}
#print(time_constraints)

# 读取df_proc：仓库数据表，获取仓库处置量上限，处置成本(单位: 万元/吨)，开仓成本（单位：万元）
df_proc = pd.read_csv(os.path.join(input_data_path, 'df_proc.csv'))
# 仓库处置量上限
warehouse_proc_upper_limits = df_proc.set_index('Name')['Capacity'].to_dict()
#print(warehouse_proc_upper_limits)

# 处置成本，单位从万元/吨调整成元/吨
warehouse_proc_costs = df_proc.set_index('Name')['Processing_fee'].to_dict()
warehouse_proc_costs = {key: val * 10000 for key, val in warehouse_proc_costs.items()}
print(warehouse_proc_costs)

# 开仓成本，单位从万元/吨调整成元/吨
warehouse_opening_costs = df_proc.set_index('Name')['Opening_fee'].to_dict()
warehouse_opening_costs = {key: val * 10000 for key, val in warehouse_opening_costs.items()}

# 读取df_order, 获取全部门店需要的进口、国产水果数量(im: 进口，dm: 国产)
df_order = pd.read_csv(os.path.join(input_data_path, 'df_order.csv'))

# ----------------------------------------------------------------------------------------------------------------------
order_demands = {name: group.set_index('SKU')['qty'].to_dict() for name, group in df_order.groupby('Name')}
#print(order_demands)

# 初始化问题实例
model = pulp.LpProblem("Minimize_Total_Cost", pulp.LpMinimize)

###
# 定义决策变量
# x_im, y_im, z_im分别表示从总仓向分仓，从总仓向门店，从分仓向门店调拨的进口(im)水果质量
x_im = pulp.LpVariable.dicts("x_im", [(cw, rw) for cw in central_warehouses for rw in regional_warehouses], lowBound=0, cat='Continuous')
y_im = pulp.LpVariable.dicts("y_im", [(cw, s) for cw in central_warehouses for s in stores], lowBound=0, cat='Continuous')
z_im = pulp.LpVariable.dicts("z_im", [(rw, s) for rw in regional_warehouses for s in stores], lowBound=0, cat='Continuous')

# x_dm, y_dm, z_dm分别表示从总仓向分仓，从总仓向门店，从分仓向门店调拨的国产(dm)水果质量
x_dm = pulp.LpVariable.dicts("x_dm", [(cw, rw) for cw in central_warehouses for rw in regional_warehouses], lowBound=0, cat='Continuous')
y_dm = pulp.LpVariable.dicts("y_dm", [(cw, s) for cw in central_warehouses for s in stores], lowBound=0, cat='Continuous')
z_dm = pulp.LpVariable.dicts("z_dm", [(rw, s) for rw in regional_warehouses for s in stores], lowBound=0, cat='Continuous')

# 仓库启用变量，0-1变量(1表示启动该仓库，0表示不启用)
# **********************************************************************************************************************
w_C = pulp.LpVariable.dicts("w_C", central_warehouses, lowBound=0, upBound=1, cat='Binary')
w_R = pulp.LpVariable.dicts("w_R", regional_warehouses, lowBound=0, upBound=1, cat='Binary')

# 购买智能调度系统的二元决策变量(1表示购买，0表示不购买)
theta_C = pulp.LpVariable.dicts("theta_C", central_warehouses, lowBound=0, upBound=1, cat='Binary')
theta_R = pulp.LpVariable.dicts("theta_R", regional_warehouses, lowBound=0, upBound=1, cat='Binary')

# 单一供应源决策变量，(1表示供应，0表示不供应)
# u_im, v_im分别表示从总仓，分仓分别向门店供应进口水果
u_im = pulp.LpVariable.dicts("u_im", [(cw, s) for cw in central_warehouses for s in stores], lowBound=0, upBound=1, cat='Binary')
v_im = pulp.LpVariable.dicts("v_im", [(rw, s) for rw in regional_warehouses for s in stores], lowBound=0, upBound=1, cat='Binary')
# u_dm, v_dm分别表示从总仓，分仓分别向门店供应国产水果
u_dm = pulp.LpVariable.dicts("u_dm", [(cw, s) for cw in central_warehouses for s in stores], lowBound=0, upBound=1, cat='Binary')
v_dm = pulp.LpVariable.dicts("v_dm", [(rw, s) for rw in regional_warehouses for s in stores], lowBound=0, upBound=1, cat='Binary')


# 定义二元变量 satisfy
satisfy = pulp.LpVariable.dicts("satisfy", [(w, s) for w in central_warehouses + regional_warehouses for s in stores],
                                lowBound=0, upBound=1, cat='Binary')


# 目标函数：总运输成本、开仓成本、处置成本和时效性惩罚
# 运输成本
transport_cost = pulp.lpSum(
    [transport_costs[cw][rw] * x_im[(cw, rw)] for cw in central_warehouses for rw in regional_warehouses] +
    [transport_costs[cw][s] * y_im[(cw, s)] for cw in central_warehouses for s in stores] +
    [transport_costs[rw][s] * z_im[(rw, s)] for rw in regional_warehouses for s in stores] +
    [transport_costs[cw][rw] * x_dm[(cw, rw)] for cw in central_warehouses for rw in regional_warehouses] +
    [transport_costs[cw][s] * y_dm[(cw, s)] for cw in central_warehouses for s in stores] +
    [transport_costs[rw][s] * z_dm[(rw, s)] for rw in regional_warehouses for s in stores]
)

# 开仓成本
warehouse_cost = pulp.lpSum(
    [warehouse_opening_costs[w] * w_C[w] for w in central_warehouses] +
    [warehouse_opening_costs[w] * w_R[w] for w in regional_warehouses]
)

# 处置成本
# handling_cost = pulp.lpSum(
#     [(1 - 0.5 * theta_C[cw]) * warehouse_proc_costs[cw] * w_C[cw] for cw in central_warehouses] +
#     [(1 - 0.5 * theta_R[rw]) * warehouse_proc_costs[rw] * w_R[rw] for rw in regional_warehouses]
# )
# 引入中间变量temp_C[cw]来处理theta_C[cw] * w_C[cw]，属于0-1变量, temp_C[cw] = theta_C[cw] * w_C[cw]


temp_C = pulp.LpVariable.dicts("temp_C", central_warehouses, lowBound=0, upBound=1, cat='Binary')
temp_R = pulp.LpVariable.dicts("temp_R", regional_warehouses, lowBound=0, upBound=1, cat='Binary')
handling_cost = pulp.lpSum(
    [warehouse_proc_costs[cw] * (w_C[cw] - 0.5 * temp_C[cw]) for cw in central_warehouses] +
    [warehouse_proc_costs[rw] * (w_R[rw] - 0.5 * temp_R[rw]) for rw in regional_warehouses]
)

# 加上智能调度系统的购买成本
smart_system_cost = pulp.lpSum(
    [single_smart_system_cost * theta_C[cw] for cw in central_warehouses] +
    [single_smart_system_cost * theta_R[rw] for rw in regional_warehouses]
)

penalty_cost = pulp.lpSum(
    [(1 - satisfy[(w, s)]) * penalty_val for w in central_warehouses + regional_warehouses for s in stores]
)

model += transport_cost + warehouse_cost + handling_cost + smart_system_cost + penalty_cost

# 定义约束条件
# 对于每个门店，进口(国产)水果只能由一个总仓或者一个分仓提供，即向门店s提供进口(国产)水果的总仓个数+分仓个数=1
for s in stores:
    model += pulp.lpSum([u_im[(cw, s)] for cw in central_warehouses] + [v_im[(rw, s)] for rw in regional_warehouses]) == 1
    model += pulp.lpSum([u_dm[(cw, s)] for cw in central_warehouses] + [v_dm[(rw, s)] for rw in regional_warehouses]) == 1

# 每个仓库调拨的质量不超过M * 单一供应源决策变量，如果决策变量为0，则该仓库不能供应
for cw in central_warehouses:
    for s in stores:
        model += y_im[(cw, s)] <= M * u_im[(cw, s)]
        model += y_dm[(cw, s)] <= M * u_dm[(cw, s)]

for rw in regional_warehouses:
    for s in stores:
        model += z_im[(rw, s)] <= M * v_im[(rw, s)]
        model += z_dm[(rw, s)] <= M * v_dm[(rw, s)]

for cw in central_warehouses:
    for s in stores:
        model += y_im[(cw, s)] <= M * w_C[cw]
        model += y_dm[(cw, s)] <= M * w_C[cw]

for rw in regional_warehouses:
    for s in stores:
        model += z_im[(rw, s)] <= M * w_R[rw]
        model += z_dm[(rw, s)] <= M * w_R[rw]

# 对于每个门店，从仓库调拨的进口(国产)水果数量不低于该门店进口(国产)水果需求量
for s in stores:
    model += pulp.lpSum([y_im[(cw, s)] for cw in central_warehouses] + [z_im[(rw, s)] for rw in regional_warehouses]
                        ) == order_demands[s]['im']
    model += pulp.lpSum([y_dm[(cw, s)] for cw in central_warehouses] + [z_dm[(rw, s)] for rw in regional_warehouses]
                        ) == order_demands[s]['dm']

# 对于总仓，调拨到分仓和门店的的水果总量不超过该仓库处置量上限
for cw in central_warehouses:
    model += pulp.lpSum(
        [x_im[(cw, rw)] for rw in regional_warehouses] + [y_im[(cw, s)] for s in stores] +
        [x_dm[(cw, rw)] for rw in regional_warehouses] + [y_dm[(cw, s)] for s in stores]
    ) <= warehouse_proc_upper_limits[cw]

# 对于分仓，调拨到门店的水果总量不超过该仓库处置量上限
for rw in regional_warehouses:
    model += pulp.lpSum(
        [z_im[(rw, s)] for s in stores] + [z_dm[(rw, s)] for s in stores]    # z_im改为z_dm
    ) <= warehouse_proc_upper_limits[rw]

for w in central_warehouses + regional_warehouses:
    for s in stores:
        model += time_constraints[w][s] <= valid_oder_time + M * (1 - satisfy[(w, s)])

# 满足率约束
total_mass = pulp.lpSum([y_im[(cw, s)] for cw in central_warehouses for s in stores] +
                        [z_im[(rw, s)] for rw in regional_warehouses for s in stores] +
                        [y_dm[(cw, s)] for cw in central_warehouses for s in stores] +
                        [z_dm[(rw, s)] for rw in regional_warehouses for s in stores]
                        )
satisfied_mass = pulp.lpSum([y_im[(cw, s)] * time_constraints[cw][s] for cw in central_warehouses for s in stores] +
                            [z_im[(rw, s)] * time_constraints[rw][s] for rw in regional_warehouses for s in stores] +
                            [y_dm[(cw, s)] * time_constraints[cw][s] for cw in central_warehouses for s in stores] +
                            [z_dm[(rw, s)] * time_constraints[rw][s] for rw in regional_warehouses for s in stores]
                            )
model += satisfied_mass >= time_satisfaction_rate * total_mass

# 对于引入的中间变量temp需要满足的约束 temp_C[cw] = theta_C[cw] * w_C[cw]
for cw in central_warehouses:
    model += temp_C[cw] <= theta_C[cw]
    model += temp_C[cw] <= w_C[cw]
    model += temp_C[cw] >= theta_C[cw] + w_C[cw] - 1

for rw in regional_warehouses:
    model += temp_R[rw] <= theta_R[rw]
    model += temp_R[rw] <= w_R[rw]
    model += temp_R[rw] >= theta_R[rw] + w_R[rw] - 1

for cw in central_warehouses:
    model += w_C[cw] == 1

# 求解模型
model.solve()

# 输出结果
print("Status:", pulp.LpStatus[model.status])
print("--------------------------------------------------------------------------------------")
print("Total Cost:", pulp.value(model.objective))
print("--------------------------------------------------------------------------------------")
print("时效满足率:", satisfied_mass.value()/total_mass.value())
print("--------------------------------------------------------------------------------------")
print("运输费用:", transport_cost.value(), "  开仓成本:", warehouse_cost.value())
print("人工成本:", handling_cost.value(), "  智能调度系统采购费用:", smart_system_cost.value())
print("时效性惩罚:", penalty_cost.value())
print("--------------------------------------------------------------------------------------")


for w in central_warehouses + regional_warehouses:
    print(f"{w} Open:", w_C[w].varValue if w in central_warehouses else w_R[w].varValue)
    print(f"{w} Smart System Purchase:", theta_C[w].varValue if w in central_warehouses else theta_R[w].varValue)

for w in central_warehouses:
    for s in stores:
        print(f"Supply {w} to {s} (Import):", y_im[(w, s)].varValue)
        print(f"Supply {w} to {s} (Local):", y_dm[(w, s)].varValue)

for rw in regional_warehouses:
    for s in stores:
        print(f"Supply {rw} to {s} (Import):", z_im[(rw, s)].varValue)
        print(f"Supply {rw} to {s} (Local):", z_dm[(rw, s)].varValue)