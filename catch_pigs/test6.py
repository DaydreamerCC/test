# # -*- coding:utf-8 -*-
# import numpy as np
# import pandas as pd
# import os
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
#
# input_data_path = r"C:\Users\ChenCong\Desktop\2024年度“火花杯”数学建模精英联赛-C题-附件\2024年度“火花杯”数学建模精英联赛-C题-附件\input_data"
#
# path_past_order = os.path.join(input_data_path, 'df_past_order.csv')
# excel_past_order = pd.read_csv(path_past_order, encoding='gbk')
# print(excel_past_order)
# print('------------------------------------------------------------------')
# print(excel_past_order.describe())
# print('------------------------------------------------------------------')
# print(excel_past_order.shape)
# print('------------------------------------------------------------------')
# print(excel_past_order.info())
# excel_past_order.dropna(inplace=True)   # 剔除空值
# excel_past_order.hist(bins=60, figsize=(15, 6))
# plt.show()




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('sales_data.csv')

# 预处理数据
data['Month'] = data['Month'].astype(int)
X = data[['Month']]
y = data['Sales']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', label='Fitted Line')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Prediction')
plt.legend()
plt.show()

# 预测未来销售额
future_months = pd.DataFrame({'Month': [7, 8, 9]})
future_sales = model.predict(future_months)
print('Future Sales Predictions:')
for month, sales in zip(future_months['Month'], future_sales):
    print(f'Month {month}: {sales:.2f}')

# import csv
# f = open('sales_data.csv', 'a', newline='', encoding='utf-8-sig')
# global writer
# writer = csv.writer(f)
# writer.writerow(['Month', 'Sales'])
# writer.writerow(['1', '200'])
# writer.writerow(['2', '220'])
# writer.writerow(['3', '250'])
# writer.writerow(['4', '275'])
# writer.writerow(['5', '300'])
# writer.writerow(['6', '320'])
