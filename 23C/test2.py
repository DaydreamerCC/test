import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

pd.set_option('display.max_columns', None)    # 运行框显示所有列
pd.set_option('display.max_rows', None)      # 运行框显示所有行
# pd.set_option('max_colwidth', 400)

# 重新加载 CSV 文件
df_past_order = pd.read_csv("C:\\Users\\ChenCong\\Desktop\\2024年度“火花杯”数学建模精英联赛-C题-附件\\2024年度“火花杯”数学建模精英联赛-C题-附件\\input_data\\df_past_order.csv")
df_loc = pd.read_csv("C:\\Users\\ChenCong\\Desktop\\2024年度“火花杯”数学建模精英联赛-C题-附件\\2024年度“火花杯”数学建模精英联赛-C题-附件\\input_data\\df_loc.csv")

# 将df_loc文件的地名加上'-shi'，确保两个文件的地名保持一致
df_loc['name'] = df_loc['name'] + '-shi'

# 可以使用0填充文件中的NaN，或者根据需求使用其他方法
df_past_order.fillna(0, inplace=True)
df_loc.fillna(0, inplace=True)

# 得到时间序列(数值型)，后面会处理成datetime
date_columns = pd.to_datetime(df_past_order.columns[2:])

# print('df_past_order: \n', df_past_order)
# print('---------------------------------------------------------')


# 重新合并数据
df_past_order_long = df_past_order.melt(id_vars=["Name", "SKU"], var_name="date", value_name="orders")
df_past_order_long['date'] = pd.to_datetime(df_past_order_long['date'])
df_past_order_long = df_past_order_long.sort_values(by=["Name", "SKU", "date"]).reset_index(drop=True)
# print('df_past_order_long: \n', df_past_order_long)
# print('---------------------------------------------------------')

df_merged = pd.merge(df_past_order_long, df_loc, left_on='Name', right_on='name', how='left')
df_merged = df_merged.drop(columns=['name'])
# print('df_merged: \n', df_merged)
# print('---------------------------------------------------------')

df_merged['day_of_week'] = df_merged['date'].dt.dayofweek
df_merged['day_of_month'] = df_merged['date'].dt.day
df_merged['month'] = df_merged['date'].dt.month
df_merged['year'] = df_merged['date'].dt.year
df_merged['is_weekend'] = df_merged['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# print('df_merged: \n', df_merged)
# print('---------------------------------------------------------')

# 2. 计算历史订单量特征
df_merged = df_merged.sort_values(by=['Name', 'SKU', 'date'])
# print('df_merged: \n', df_merged)
# print('---------------------------------------------------------')
df_merged['orders_last_7_days'] = df_merged.groupby(['Name', 'SKU'])['orders'].transform(lambda x: x.rolling(window=7).mean())

# 3. 编码城市名称
le = LabelEncoder()
df_merged['Name_encoded'] = le.fit_transform(df_merged['Name'])

# print('df_merged: \n', df_merged)
# print('---------------------------------------------------------')

# 准备训练数据
features = ['Name_encoded', 'SKU', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend',
            'Longitude', 'Latitude', 'city_area', 'builtup_area', 'resident_pop', 'gdp', 'orders_last_7_days']

X = df_merged[features]
X['SKU'] = X['SKU'].apply(lambda x: 1 if x == 'dm' else 0)  # 如果SKU为"dm"则为1，否则为0
# print('X: \n', X)
# print('---------------------------------------------------------')
y = df_merged['orders']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# print('X_train: \n', X_train)
# print('---------------------------------------------------------')
# print('y_train: \n', y_train)
# print('---------------------------------------------------------')
# print('X_test: \n', X_test)
# print('---------------------------------------------------------')
# print('y_test: \n', y_test)
# print('---------------------------------------------------------')





# 设定参数分布
param_distributions = {
    'n_estimators': randint(100, 1000),  # 弱学习器（树）的数量
    'learning_rate': uniform(0.01, 0.1),  # 学习率
    'max_depth': randint(3, 12),  # 树的最大深度
}

# 初始化模型
model = XGBRegressor(random_state=48)

# 随机搜索模型
random_search = RandomizedSearchCV(
    estimator=model,  # 要优化的模型
    param_distributions=param_distributions,  # 参数分布
    n_iter=50,  # 随机搜索的迭代次数
    scoring='neg_mean_absolute_error',  # 评价指标，回归中一般使用MSE
    cv=3,  # 3折交叉验证
    verbose=1,  # 显示进度
    random_state=48,  # 保证结果可复现
    n_jobs=-1  # 使用所有CPU核
)

# 进行搜索，X_train和y_train为训练数据
random_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", random_search.best_params_)





# # 4. 训练XGBoost模型
# model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10, random_state=42)
# model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred)
y_pred.index = y_test.index
print('X_test: \n', X_test)
print('---------------------------------------------------------')
# print('y_test: \n', y_test)
# print('---------------------------------------------------------')
# print('y_pred: \n', y_pred)
# print('---------------------------------------------------------')

df = pd.concat([y_test, y_pred], axis=1)
df.columns = ['y_test', 'y_pred']
print('df: \n', df)
print('---------------------------------------------------------')

mae = mean_absolute_error(y_test, y_pred)
print('mae = ', mae)
print('---------------------------------------------------------')