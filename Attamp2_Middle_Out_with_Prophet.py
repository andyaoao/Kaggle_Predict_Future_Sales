import numpy as np
import pandas as pd
import random as rd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time


# 時系列関連library
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from fbprophet import Prophet

# データを取り込む
sales=pd.read_csv("./Datasets/PredictFutureSales/sales_train.csv")
item_cat=pd.read_csv("./Datasets/PredictFutureSales/item_categories.csv")
item=pd.read_csv("./Datasets/PredictFutureSales/items.csv")
sub=pd.read_csv("./Datasets/PredictFutureSales/sample_submission.csv")
shops=pd.read_csv("./Datasets/PredictFutureSales/shops.csv")
test=pd.read_csv("./Datasets/PredictFutureSales/test.csv")

# データ型の確認
# print (sales.info())

# 日付データ型を整理
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# 前月のデータを抽出
sales_201510 = sales[sales["date_block_num"] == 33]
# as_index=Falseは実テーブルのように、一レコード一行で格納
# 2015/10の商品別、店舗別の集計
sales_201510 = sales_201510.groupby(["shop_id", "item_id"])["item_cnt_day"].sum()

print ("sales_201510")
print (sales_201510.head())

sales_201510 = sales_201510.unstack()
sales_201510 = sales_201510.fillna(0)
print ("sales_201510_shop")
print (sales_201510.head())

# 比率を計算する
sales_201510 = sales_201510.apply(lambda x: x/sum(x))
print ("sales_201510_percentage")
print (sales_201510.head())

sales_201510 = sales_201510.stack()
sales_201510 = sales_201510.reset_index()
sales_201510.columns=['shop_id','item_id','item_cnt_day']
print ("sales_201510_big_table")
print (sales_201510.head())

# 店舗別の売上データを整理する
monthly_shop_sales=sales.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()
# unstack関数で、行を列に変換(level 1 )
monthly_shop_sales=monthly_shop_sales.unstack(level=1)
monthly_shop_sales=monthly_shop_sales.fillna(0)
dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
monthly_shop_sales.index=dates
monthly_shop_sales=monthly_shop_sales.reset_index()
print ("monthly_shop_sales")
print (monthly_shop_sales.head())


# 店舗ごとに時系列データを作る
print ("length of sales is {}".format(len(monthly_shop_sales)))
forecastsDict = {}
for node in range(len(monthly_shop_sales)):

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)
    # print ("nodeToForecast")
    # print (nodeToForecast.head())
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    # mを投入したのは、20130101-20151001の期間
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    # futureの出力は、20130101-20151101の期間(periodは1のため、元の系列に1期間追加)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)
    # m.plot(forecastsDict[node])
    # plt.show()

# 全店舗の予測を一つのdfに整理
predictions = np.zeros([len(forecastsDict[0].yhat),1])
# print ("predictions")
# print (predictions)

# 全てのcolumn数は、
nCols = len(list(forecastsDict.keys()))+1
# print ("nCols")
# print (nCols)

for key in range(0, nCols-1):
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:, np.newaxis]
    if key==0:
        predictions=f2.copy()
       # print(predictions.shape)
    else:
       predictions = np.concatenate((predictions, f2), axis = 1)


predictions_after=predictions[-1]
predictions_df = pd.DataFrame({'forecast':predictions_after})
print ("prediction after")
print (predictions_df)

predictions_df["shop_id"] = predictions_df.index + 1
print ("predictions_df")
print (predictions_df)

# percentage calculation
sales_201510 = pd.merge(sales_201510, predictions_df, left_on=["shop_id"], right_on=["shop_id"], how='left')
print ("after merge")
print (sales_201510.head())

sales_201510["item_cnt_month"] = sales_201510["item_cnt_day"] * sales_201510["forecast"]
sales_201510["item_cnt_month"] = sales_201510["item_cnt_month"].map(lambda x : 0 if x < 0 else x)
print ("sales_201510")
print (sales_201510.head())

submission = pd.merge(test, sales_201510, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how='left')
print ("submission before delete")
print (submission.head())

submission_new = submission.drop(['shop_id', 'item_id','forecast','item_cnt_day'], axis=1)
submission_new = submission_new.fillna(0)
print ("submission after delete")
print (submission_new.head())

submission_new.columns=['ID','item_cnt_month']
print ("submission")
print (submission_new.head())

# csvに書き出し
submission_new.to_csv("./Datasets/PredictFutureSales/submission.csv", index=False)
