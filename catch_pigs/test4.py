import pandas as pd
import pulp
#
# path_matrix = "C:\\Users\\ChenCong\\Desktop\\df_matrix.csv"
# path_order = "C:\\Users\\ChenCong\\Desktop\\df_order.csv"
# path_proc = "C:\\Users\\ChenCong\\Desktop\\df_proc.csv"
#
# excel_matrix = pd.read_csv(path_matrix, encoding='gbk')
# excel_order = pd.read_csv(path_order, encoding='gbk')
# excel_proc = pd.read_csv(path_proc, encoding='gbk')
#
# # delta_RDC_shop =
# # delta_CDC_shop
#
# prob = pulp.LpProblem('myProb', sense=pulp.LpMinimize)
#
# x = pulp.LpVariable.dicts("x", ((i, j) for i in range(1) for j in range(18)), cat='Continuous', lowBound=0)
# y = pulp.LpVariable.dicts("y", ((j, k) for j in range(18) for k in range(106)), cat='Continuous', lowBound=0)
# z = pulp.LpVariable.dicts("z", ((i, k) for i in range(1) for k in range(106)), cat='Continuous', lowBound=0)
#
# p = pulp.LpVariable.dicts("p", ((i, j) for i in range(1) for j in range(18)), cat='Continuous', lowBound=0)
# q = pulp.LpVariable.dicts("q", ((j, k) for j in range(18) for k in range(106)), cat='Continuous', lowBound=0)
# m = pulp.LpVariable.dicts("m", ((i, k) for i in range(1) for k in range(106)), cat='Continuous', lowBound=0)
#
# u = pulp.LpVariable.dicts("u", ((j, k) for j in range(18) for k in range(106)), cat='Binary')
# v = pulp.LpVariable.dicts("v", ((i, k) for i in range(1) for k in range(106)), cat='Binary')
# open = pulp.LpVariable.dicts("open", ((i, k) for i in range(1) for k in range(106)), cat='Binary')


# from pulp import LpVariable, LpMaximize, LpProblem, value, LpStatus, LpMinimize
#
# # 第一个参数为这个问题取名字，第二个参数表示求目标函数的最大值（LpMinimize）
# prob = LpProblem('max_z', sense=LpMinimize)
# # name为变量名， lowBound为下边界，None为没边界
# x = LpVariable(name='x', lowBound=0, upBound=None)
# y = LpVariable('y', lowBound=0, upBound=None)
# z = LpVariable('z', lowBound=0, upBound=5)
#
# # 设置目标函数
# prob += 2*x+3*y-5*z
# # 约束条件
# #prob += x+y+z == 7
#
#
# status = prob.solve()
# print("求解状态:", LpStatus[prob.status])
# print(f"目标函数的最大值z={value(prob.objective)}，此时目标函数的决策变量为:",
#       {v.name: v.varValue for v in prob.variables()})

list1 = [1, 2, 3]
list2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
list3 = [list1[i] * list2[i][j] for i in range(len(list1)) for j in range(3)]
print(list3)
