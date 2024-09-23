import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('max_colwidth', 400)


# 读取Excel文件
file_path = "E:\\2022C题\\C题\\附件.xlsx"
df_sheet1 = pd.read_excel(file_path, sheet_name=0)
df_sheet2 = pd.read_excel(file_path, sheet_name=1)
df_sheet3 = pd.read_excel(file_path, sheet_name=2)

# print('df_sheet2: \n', df_sheet2)
#
# # 从表单1中提取我们需要的字段：风化情况和玻璃类型
# df_relevant = df_sheet1[['表面风化', '类型']]
#
# # 创建交叉表
# contingency_table_actual = pd.crosstab(df_relevant['表面风化'], df_relevant['类型'])
#
# # 进行卡方检验
# chi2_actual, p_actual, dof_actual, expected_actual = chi2_contingency(contingency_table_actual)
#
# # 输出卡方检验结果
# result_actual = {
#     'chi2': chi2_actual,
#     'p-value': p_actual,
#     'degrees of freedom': dof_actual,
#     'expected frequencies': expected_actual
# }
#
# print('contingency_table_actual: \n', contingency_table_actual)
# print('----------------------------------------------')
# print('result_actual: \n', result_actual)


# 提取表单2中的采样点编号部分（只保留编号，不考虑部位）
df_sheet2['文物编号'] = df_sheet2['文物采样点'].str.extract('(\d+)').astype(int)

# 合并表单1和表单2，依据文物编号进行关联
df_merged = pd.merge(df_sheet1, df_sheet2, on='文物编号')

# 分组：根据玻璃类型和风化情况对化学成分进行分组统计
# 选取关键化学成分
components = ['二氧化硅(SiO2)', '氧化钠(Na2O)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化镁(MgO)',
              '氧化铝(Al2O3)', '氧化铁(Fe2O3)', '氧化铜(CuO)', '氧化铅(PbO)', '氧化钡(BaO)']

# 计算每个玻璃类型和风化情况的化学成分的均值和标准差
grouped = df_merged.groupby(['类型', '表面风化'])[components].agg(['mean', 'std'])
print('grouped: \n', grouped)