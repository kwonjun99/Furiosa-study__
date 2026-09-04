#https://www.kaggle.com/competitions/bike-sharing-demand/rules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


#1. data
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

#print(train_csv.columns) #컬럼명들이 나옴
#print(train_csv.info()) #데이터 정보를 가져옴 -> 결측치 없음을 확인 바로진행
#print(train_csv.describe()) #데이터 정보 자세히 묘사

############ 결측치 확인 #############
print(train_csv.isna().sum()) #->결측치 있는지 확인하는 코드 이게 더 편해보이네?


x = train_csv.drop(['casual','registered','count'], axis=1) #축 행:0 열:1 count라는 열 삭제
print(x) #[10886 rows x 8 columns]

y = train_csv['count']
print(y, y.shape)  #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=48
)


#2. model
model = Sequential()
model.add(Dense(512, input_dim=8))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. compile, train
model.compile(loss='mse', optimizer='adam')
start_time = time.time()

batch_size = 3
history = model.fit(x_train,y_train, epochs=358, batch_size = 3)

train_time = time.time() - start_time

#4. evaluate, predict
loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)
submission['count'] = y_submit

submission.to_csv(path + "submit/" + "submit_0904_1655.csv", index=True)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 48,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2 = r2
)