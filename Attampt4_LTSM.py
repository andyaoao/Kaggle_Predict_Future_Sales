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

# LSTM関連library
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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
sales = sales.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)
sales = sales.reset_index()
print ("sales")
print (sales.head())

sales = pd.merge(test, sales, on=['item_id', 'shop_id'], how='left')
sales = sales.fillna(0)
print ("sales merge with test")
print (sales.head())

sales = sales.drop(['shop_id', 'item_id', 'ID'], axis=1)
sales = sales.fillna(0)
print ("sales preparation")
print (sales.head())

print ("sales.values")
print (sales.values)
print (sales.values[:,:-1])
X_train = np.expand_dims(sales.values[:, :-1], axis=2)
y_train = sales.values[:, -1:]

X_test = np.expand_dims(sales.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)

model = Sequential()
model.add(LSTM(units=64, input_shape=(33, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])
print (model.summary())

history = model.fit(X_train, y_train, batch_size=4096, epochs=10)

plt.plot(history.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(history.history['mean_squared_error']), label= 'rmse')
plt.legend(loc=1)
plt.show()

LSTM_prediction = model.predict(X_test)

submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': LSTM_prediction.ravel()})
submission.to_csv('./Datasets/PredictFutureSales/submission.csv',index=False)
