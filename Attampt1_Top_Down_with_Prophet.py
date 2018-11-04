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

# 前年度の11月データを抽出
sales_201411 = sales[sales["date_block_num"] == 22]
# as_index=Falseは実テーブルのように、一レコード一行で格納
sales_201411 = sales_201411.groupby(["date_block_num","shop_id","item_id"], as_index=False).sum()
print ("sales_201411")
print (sales_201411.head())

# 比率を計算する
# sum関数で全体のpercentageを算出
sales_201411["percentage"] = sales_201411["item_cnt_day"] / sales_201411["item_cnt_day"].sum()
print ("sales_201411 percentage")
print (sales_201411.head())

# date block num(年月)をベースで全社の販売点数を積上げる
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
print ("ts")
print (ts.head())

# prophetが受け入れるデータ形は、日付(ds)と値(y)
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
# 列名を修正する
ts.columns=['ds','y']
print ("before modeling")
print (ts.head())

#時系列モデルを定義
# パラメータは、年周期があること
model = Prophet('linear', yearly_seasonality=True)
model.fit(ts)

# 2017/11を予測
future = model.make_future_dataframe(periods = 1, freq = 'MS')
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print ("forecast")
print (forecast.head())

# 2018/11は最後期
forecast_value = forecast['yhat'].values[-1]
sales_201411["result"] = sales_201411["percentage"] * forecast_value
print ("calculation")
print (sales_201411.head())

submission = pd.merge(test, sales_201411, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how='left')
print ("submission before delete")
print (submission.head())
submission_new = submission.drop(['date_block_num', 'shop_id', 'item_id','item_price','item_cnt_day', 'percentage'], axis=1)
submission_new = submission_new.fillna(0)
print ("submission after delete")
print (submission_new.head())

submission_new.columns=['ID','item_cnt_month']
print ("submission")
print (submission_new.head())

# csvに書き出し
submission_new.to_csv("./Datasets/PredictFutureSales/submission.csv", index=False)

# 2015/10の売上配分を計算。
# 予測を配分。
