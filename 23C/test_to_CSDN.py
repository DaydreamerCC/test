import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 输入至函数predict_sarimax()中的series的index和values类型:
# series.index = <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
# series.values = <class 'numpy.ndarray'>


def predict_sarimax(series, exog, steps=7):
    model = SARIMAX(series, exog=exog, order=(5, 1, 0))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps, exog=exog[-steps:])
    return forecast


plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

pd.set_option('display.max_columns', None)    # 运行框显示所有列
pd.set_option('display.max_rows', None)      # 运行框显示所有行
pd.set_option('max_colwidth', 400)

# 重新加载 CSV 文件
df_past_order = pd.read_csv("C:\\Users\\ChenCong\\Desktop\\2024年度“火花杯”数学建模精英联赛-C题-附件\\2024年度“火花杯”数学建模精英联赛-C题-附件\\input_data\\df_past_order.csv")
df_loc = pd.read_csv("C:\\Users\\ChenCong\\Desktop\\2024年度“火花杯”数学建模精英联赛-C题-附件\\2024年度“火花杯”数学建模精英联赛-C题-附件\\input_data\\df_loc.csv")

# 将df_loc文件的地名加上'-shi'，确保两个文件的地名保持一致
df_loc['name'] = df_loc['name'] + '-shi'

# 可以使用0填充文件中的NaN，或者根据需求使用其他方法
df_past_order.fillna(0, inplace=True)
df_loc.fillna(0, inplace=True)

# 得到时间序列(数值型)，后面会处理成datetime
date_columns = df_past_order.columns[2:]

# 遍历每个城市和SKU进行预测
predictions = []

for city in df_past_order['Name'].unique():  # 遍历每个城市
    for item in ['im', 'dm']:  # 对两种货物分别处理 ('im' 和 'dm')
        # 获取特定city和特定item的商品历史订单量，并将其处理成series，index设置成to_datetime(date_columns)
        series = (df_past_order[(df_past_order['Name'] == city) & (df_past_order['SKU'] == item)])[date_columns]
        series = series.stack()
        series.index = pd.to_datetime(date_columns)

        # 输入至predict_sarimax(series, exog, steps=7)中的series和exog的时间序列(index)要保持一致
        exog = df_loc[df_loc['name'] == city][['Longitude', 'Latitude', 'city_area', 'builtup_area', 'resident_pop', 'gdp']]
        exog = np.tile(exog, (len(date_columns), 1))    # exog初始只有一行，将其复制成len(date_columns)行
        exog = pd.DataFrame(exog)
        exog.columns = ['Longitude', 'Latitude', 'city_area', 'builtup_area', 'resident_pop', 'gdp']
        exog.index = pd.to_datetime(date_columns)

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

    # 数据点太多，只打印五个观察
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