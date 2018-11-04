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

# 商品別、店舗別、月別の集計
sales = sales.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].sum()
sales = sales.unstack().unstack()
sales = sales.fillna(0)
print ("sales")
print (sales.head())

dates = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
sales.index=dates
sales = sales.reset_index()
print ("sales")
print (sales.head())

print ("length of sales is {}".format(len(sales.columns)))
forecastsDict = {}
for node in range(len(sales.columns)):
# for node in range(3):

    nodeToForecast = pd.concat([sales.iloc[:,0], sales.iloc[:, node+1]], axis = 1)
    # print ("nodeToForecast")
    # print (nodeToForecast.head())
    # print (nodeToForecast.shape)
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    if node < 1 :
        print ("nodeToForecast")
        print (nodeToForecast)
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)

nCols = len(list(forecastsDict.keys()))+1
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
sales = pd.merge(sales, predictions_df, left_on=["shop_id"], right_on=["shop_id"], how='left')
print ("after merge")
print (sales.head())

sales["item_cnt_month"] = sales["item_cnt_day"] * sales["forecast"]
sales["item_cnt_month"] = sales["item_cnt_month"].map(lambda x : 0 if x < 0 else x)
print ("sales")
print (sales.head())

submission = pd.merge(test, sales, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how='left')
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
