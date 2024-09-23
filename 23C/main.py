import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# series.index = <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
# series.values = <class 'numpy.ndarray'>


def predict_sarimax(series, exog, steps=7):
    model = SARIMAX(series, exog=exog, order=(5, 1, 0))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps, exog=exog[-steps:])
    return forecast


plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('max_colwidth', 400)

# 重新加载 CSV 文件
df_past_order = pd.read_csv("C:\\Users\\ChenCong\\Desktop\\2024年度“火花杯”数学建模精英联赛-C题-附件\\2024年度“火花杯”数学建模精英联赛-C题-附件\\input_data\\df_past_order.csv")
df_loc = pd.read_csv("C:\\Users\\ChenCong\\Desktop\\2024年度“火花杯”数学建模精英联赛-C题-附件\\2024年度“火花杯”数学建模精英联赛-C题-附件\\input_data\\df_loc.csv")

# # 去除 'Name' 列中末尾的 'shi' 和 '-'
# df_past_order['Name'] = df_past_order['Name'].str.replace('shi$', '', regex=True).str.rstrip('-')
df_loc['name'] = df_loc['name'] + '-shi'

df_past_order.fillna(0, inplace=True)  # 可以使用0填充NaN，或者根据需求使用其他方法
df_loc.fillna(0, inplace=True)  # 可以使用0填充NaN，或者根据需求使用其他方法

# print('df_past_order: ', df_past_order)
# print('-------------------------------------------')
# print('df_loc: ', df_loc)
#
date_columns = df_past_order.columns[2:]
# print('date_columns: ', date_columns)
# print('------------------------------------------------')
# print('list date_columns: ', list(date_columns))
# print('------------------------------------------------')

# 遍历每个城市和SKU进行预测
predictions = []

for city in df_past_order['Name'].unique():  # 遍历每个城市
    for item in ['im', 'dm']:  # 对两种货物分别处理 ('im' 和 'dm')
        series = (df_past_order[(df_past_order['Name'] == city) & (df_past_order['SKU'] == item)])[date_columns]
        # print('series[date_columns]: ', series[date_columns])
        # print('------------------------------------------------')
        series = series.stack()
        # series.index = date_columns
        series.index = pd.to_datetime(date_columns)
        # series = series.asfreq('M')
        # print('series: \n', series)
        # print('------------------------------------------------')
        # print('series type: ', type(series))
        # print('------------------------------------------------')
        # print('series index type: ', type(series.index))
        # print('------------------------------------------------')
        # print('series columns ', series.columns)
        # print('------------------------------------------------')

        exog = df_loc[df_loc['name'] == city][['Longitude', 'Latitude', 'city_area', 'builtup_area', 'resident_pop', 'gdp']]
        # print('exog: ', exog)
        # print('------------------------------------------------')
        exog = np.tile(exog, (len(date_columns), 1))
        # print('new exog: ', exog)
        # print('------------------------------------------------')
        exog = pd.DataFrame(exog)
        exog.columns = ['Longitude', 'Latitude', 'city_area', 'builtup_area', 'resident_pop', 'gdp']
        exog.index = pd.to_datetime(date_columns)

        # print('new new exog: \n', exog)
        # print('------------------------------------------------')

        forecast = predict_sarimax(series, exog, steps=7)

        prediction = {'Name': city, 'SKU': item}
        for i, val in enumerate(forecast):
            prediction[f'forecast_day_{i + 1}'] = val
        predictions.append(prediction)

# 将预测结果转换为DataFrame格式
df_predictions = pd.DataFrame(predictions)
df_predictions.to_excel('df_predictions.xlsx')
# 绘制每个城市的未来7天预测数据

# 设置图形大小
plt.figure(figsize=(12, 8))

# 遍历每个城市的预测数据
for i, row in df_predictions.iterrows():
    days = [f'forecast_day_{j + 1}' for j in range(7)]  # 创建未来7天的标签
    values = row[days].values  # 获取对应的预测值

    plt.plot(days, values, marker='o', label=f'{row["Name"]} ({row["SKU"]})')

    if i == 5:
        break

# 添加标题和标签
plt.title('未来7天的订单预测')
plt.xlabel('预测日期')
plt.ylabel('预测订单量')
plt.legend(loc='upper right')
plt.grid(True)

# 显示图形
plt.show()