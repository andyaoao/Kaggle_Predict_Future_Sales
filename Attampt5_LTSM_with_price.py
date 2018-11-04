import time
import datetime

import numpy as np
import pandas as pd

# 機械学習系のlibrary
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import gc

# Viz
import matplotlib.pyplot as plt

# データを取り込む
sales=pd.read_csv("./Datasets/PredictFutureSales/sales_train.csv", parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
item_cat=pd.read_csv("./Datasets/PredictFutureSales/item_categories.csv")
item=pd.read_csv("./Datasets/PredictFutureSales/items.csv")
sub=pd.read_csv("./Datasets/PredictFutureSales/sample_submission.csv")
shops=pd.read_csv("./Datasets/PredictFutureSales/shops.csv")
test=pd.read_csv("./Datasets/PredictFutureSales/test.csv")

# データ修正
sales[sales['item_id'] == 11373][['item_price']].sort_values(['item_price'])
sales[sales['item_id'] == 11365].sort_values(['item_price'])

# Correct sales values
sales['item_price'][2909818] = np.nan
sales['item_cnt_day'][2909818] = np.nan
sales['item_price'][2909818] = sales[(sales['shop_id'] ==12) & (sales['item_id'] == 11373) & (sales['date_block_num'] == 33)]['item_price'].median()
sales['item_cnt_day'][2909818] = round(sales[(sales['shop_id'] ==12) & (sales['item_id'] == 11373) & (sales['date_block_num'] == 33)]['item_cnt_day'].median())
sales['item_price'][885138] = np.nan
sales['item_price'][885138] = sales[(sales['item_id'] == 11365) & (sales['shop_id'] ==12) & (sales['date_block_num'] == 8)]['item_price'].median()

# 店舗商品ごとの売上、点数(合計)
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
print ("df")
print (df.head())
df = df[['date','item_id','shop_id','item_cnt_day']]
df["item_cnt_day"].clip(0.,20.)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
print ("df after clip")
print (df.head())

# 点数とテストセットの形に整理する
test_sales = pd.merge(test,df,on=['item_id','shop_id'], how='left').fillna(0)
test_sales = test_sales.drop(labels=['ID','item_id','shop_id'],axis=1)

# 店舗商品ごとの売上（平均）
scaler = MinMaxScaler(feature_range=(0, 1))
sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()
print ("df2")
print (df2.head())

# 売上とテストセットの形に整理する
price = pd.merge(test,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

print ("test_sales")
print (test_sales)

# training set の　y は　最新の店舗商品点数データとする
y_train = test_sales["2015-10"]
print ("y_train")
print (y_train)

# training set の x は　最新以外のデータ
x_sales = test_sales.drop(labels=['2015-10'],axis=1)

# dataframeをarrayにrechapeする
# 行は店舗商品数214200、列は時間軸33、階数は1(後ほどpriceを追加したいから、3Dを定義する)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
print ("x_sales")
print (x_sales)

x_prices = price.drop(labels=['2015-10'],axis=1)
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
print ("x_price")
print (x_prices)

# 点数と売上のarrayを繋げる
X = np.append(x_sales,x_prices,axis=2)
print ("X")
print (X)

# training set の y もnumpy のarrayにreshapeする
y = y_train.values.reshape((214200, 1))
print("Training Predictor Shape: ",X.shape)
print("Training Predictee Shape: ",y.shape)
del y_train, x_sales; gc.collect()

# 予測用の set をnumpy のarrayにreshapeする
# 最初の日付をtesting set から除外(予測用のdata setは同じ時間範囲を持たせるため)
test_sales = test_sales.drop(labels=['2013-01'],axis=1)
x_test_sales = test_sales.values.reshape((test_sales.shape[0], test_sales.shape[1], 1))
x_test_prices = price.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))

# 予測用のsetも点数と売上を結合
X_test = np.append(x_test_sales,x_test_prices,axis=2)
del x_test_sales,x_test_prices, price; gc.collect()
print("Test Predictor Shape: ",X_test.shape)

print("Modeling Stage")
# Define the model layers
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model_lstm.add(Dropout(0.5))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])
print(model_lstm.summary())

# Train Model
print("\nFit Model")
VALID = True
LSTM_PARAM = {"batch_size":128,
              "verbose":2,
              "epochs":10}

if VALID is True:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=False)
    # del X,y; gc.collect()
    print("X Train Shape: ",X_train.shape)
    print("X Valid Shape: ",X_valid.shape)
    print("y Train Shape: ",y_train.shape)
    print("y Valid Shape: ",y_valid.shape)

    callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]
    hist = model_lstm.fit(X_train, y_train,
                          validation_data=(X_valid, y_valid),
                          callbacks=callbacks_list,
                          **LSTM_PARAM)
    pred = model_lstm.predict(X_test)

    # Model Evaluation
    best = np.argmin(hist.history["val_loss"])
    print("Optimal Epoch: {}",best)
    print("Train Score: {}, Validation Score: {}".format(hist.history["loss"][best],hist.history["val_loss"][best]))

    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Square Error")
    plt.legend()
    plt.show()
    plt.savefig("Train and Validation MSE Progression.png")

if VALID is False:
    print("X Shape: ",X.shape)
    print("y Shape: ",y.shape)
    hist = model_lstm.fit(X,y,**LSTM_PARAM)
    pred = model_lstm.predict(X)

    plt.plot(hist.history['loss'], label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Square Error")
    plt.legend()
    plt.show()
    plt.savefig("Training Loss Progression.png")

print("\Output Submission")
submission = pd.DataFrame(pred,columns=['item_cnt_month'])
submission.to_csv('./Datasets/PredictFutureSales/submission.csv',index_label='ID')
print(submission.head())
