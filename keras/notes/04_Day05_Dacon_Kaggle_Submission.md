# Day 05 - Dacon/Kaggle Submission, Pandas 전처리, 회귀 평가, Git 동기화

## 1. Day 05 핵심

Day 05에서는 지금까지 배운 Keras Dense 회귀 모델을 실제 Dacon/Kaggle 데이터 구조에 적용하고, 최종 `submission.csv`까지 생성하는 흐름을 학습했다.

```text
CSV 데이터
↓
Pandas로 데이터 확인
↓
결측치 처리
↓
x / y 분리
↓
train_test_split
↓
Keras Dense 모델 생성
↓
model.fit()
↓
MSE / RMSE / R² 평가
↓
대회 test.csv 예측
↓
submission 파일 생성
↓
Dacon / Kaggle 제출
```

현재는 **Dense 기반 회귀 모델 + 기본 데이터 전처리 + 평가 + 제출 과정**까지 학습한 단계다.

---

## 2. MSE / MAE / RMSE / R²

### MSE
MSE = Mean Squared Error, 평균제곱오차.

- 실제값과 예측값의 차이를 제곱한 뒤 평균
- 큰 오차에 더 큰 패널티
- Keras 회귀 loss로 자주 사용

```python
model.compile(loss='mse', optimizer='adam')
```

### MAE
MAE = Mean Absolute Error, 평균절대오차.

- 실제값과 예측값 차이의 절대값 평균
- MSE보다 큰 오차의 영향이 상대적으로 작음

```python
from sklearn.metrics import mean_absolute_error
```

### RMSE
RMSE는 MSE에 square root를 적용한 값.

```python
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
```

장점:

> **원래 y와 같은 단위로 오차를 해석할 수 있다.**

### R²
R²는 회귀 모델이 데이터의 변화를 얼마나 잘 설명하는지 보는 평가 지표.

```python
r2 = r2_score(y_test, y_predict)
```

대략적으로:

```text
R² → 1에 가까울수록 좋음
R² → 0이면 평균값 수준
R² < 0이면 평균값 예측보다도 좋지 않을 수 있음
```

---

## 3. Python 문자열과 경로

Python에서 큰따옴표와 작은따옴표는 모두 문자열에 사용할 수 있다.

```python
"a"
'a'
```

### 상대경로

```python
path = "./_data/ddarung/"
```

여기서 `.`은 **현재 작업 디렉터리(Current Working Directory)** 를 의미한다.

예를 들어 현재 실행 위치가 `C:\study`라면:

```text
./_data/ddarung/
```

은:

```text
C:\study\_data\ddarung\
```

을 의미한다.

> `.`이 항상 `study`라는 뜻은 아니다. 어디서 실행했는지에 따라 기준이 달라진다.

### 절대경로

Windows:

```python
path = "C:/study/_data/ddarung/"
```

macOS:

```python
path = "/Users/username/study/_data/ddarung/"
```

Windows에서도 Python 경로는 `/`를 쓰는 편이 편하다. `\`는 escape 문자와 충돌할 수 있다.

---

## 4. 주석

한 줄 주석:

```python
# 주석
```

여러 줄 설명에 triple quote를 사용할 수도 있다.

```python
"""
여러 줄 설명
"""
```

단, triple quote는 엄밀히 말하면 **문자열 리터럴**이며 Python의 실제 주석 문법은 `#`이다.

---

## 5. Feature / Column / Attribute

표 형태 데이터에서는 다음 표현이 비슷한 의미로 자주 사용된다.

```text
열
Column
Feature
속성(Attribute)
```

예:

```text
hour
temperature
humidity
windspeed
```

등이 모델의 입력 feature가 된다.

---

## 6. 대회의 기본 데이터 구조

Dacon/Kaggle 지도학습 대회에서는 보통 다음 파일을 제공한다.

```text
train.csv
test.csv
submission.csv
```

### train.csv
정답 `y`가 존재한다.

```python
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
```

### test.csv
대회에서 제공하는 실제 시험 데이터다.

```text
x만 존재
y는 숨겨져 있음
```

학습 완료 후:

```python
y_submit = model.predict(test_csv)
```

으로 정답을 예측한다.

### submission.csv
대회에서 요구하는 제출 형식이다.

```python
submission['count'] = y_submit
submission.to_csv("submission_result.csv")
```

---

## 7. x_test와 대회 test.csv 차이

```text
train.csv
↓
x / y 분리
↓
train_test_split
↓
x_train / x_test
y_train / y_test
```

여기의 `x_test`, `y_test`는 **내가 모델 성능을 확인하기 위해 만든 내부 평가 데이터**다.

반면:

```text
대회 test.csv
```

는 정답이 숨겨진 **실제 제출용 시험 데이터**다.

```text
x_test = 모의고사
대회 test.csv = 실제 시험
```

---

## 8. 결측치 처리

결측치 확인:

```python
train_csv.isna().sum()
```

또는:

```python
train_csv.isnull().sum()
```

### Train 데이터
현재 수업에서는 가장 단순한 방식으로 결측치가 있는 행을 삭제했다.

```python
train_csv = train_csv.dropna()
```

### Test 데이터
제출용 `test.csv`는 행을 삭제하면 제출 개수가 달라지므로 `dropna()`를 사용하면 안 된다.

현재 수업에서는 평균값으로 채웠다.

```python
test_csv = test_csv.fillna(test_csv.mean())
```

---

## 9. 현재 Epoch와 Weight

현재 수업 코드는 정해진 epoch까지 학습한 마지막 weight를 사용한다.

```python
history = model.fit(
    x_train,
    y_train,
    epochs=380,
    batch_size=5
)
```

향후에는 validation loss를 이용해 가장 좋은 시점의 weight를 저장/복원하는 방법을 배우게 된다.

```text
EarlyStopping
ModelCheckpoint
```

현재는 개념만 기억한다.

---

## 10. Public / Private Leaderboard

Kaggle/Dacon 대회에서는 leaderboard 평가 데이터를 나누기도 한다.

```text
Public Leaderboard
Private Leaderboard
```

- Public: 대회 중 일부 test 정답 기준으로 보여주는 점수
- Private: 최종 순위를 결정하기 위한 별도 숨겨진 test 정답 기준 점수

따라서 Public 점수에만 지나치게 맞추면 Private에서 성능이 떨어질 수 있다.

---

## 11. AI 실력 단계 메모

```text
Level 1
모델 사용할 줄 앎

Level 2
데이터 넣고 학습/평가 가능

Level 3
성능이 안 나올 때 원인 분석 가능

Level 4
모델 / 데이터 / 평가 지표 직접 설계 가능

Level 5
배포 / 최적화 / 시스템 통합 가능

Level 6
논문 수준의 개선 / 새로운 방법 제안 가능
```

현재는 Level 1~2의 기본기를 쌓는 단계다.

---

## 12. Dacon 따릉이 정리 코드

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import time
import my_util


# 1. DATA
path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=79
)


# 2. MODEL
model = Sequential()
model.add(Dense(256, input_dim=9))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))


# 3. COMPILE / TRAIN
model.compile(loss='mse', optimizer='adam')

start_time = time.time()

batch_size = 5

history = model.fit(
    x_train,
    y_train,
    epochs=380,
    batch_size=batch_size
)

train_time = time.time() - start_time


# 4. EVALUATE
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)


# 5. DACON TEST DATA
test_csv = test_csv.fillna(test_csv.mean())

y_submit = model.predict(test_csv).flatten()

submission['count'] = y_submit

print(submission.head())
print(submission.isnull().sum())

submission.to_csv(
    path + "submission_result.csv",
    index=True
)


# 6. MODEL HISTORY
my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=79,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=loss,
    r2=r2
)
```

---

## 13. Kaggle Bike Sharing Demand 정리 코드

현재 단계에서는 Dense 회귀 모델을 이용해 기본 제출 구조를 연습한다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import my_util

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# 1. DATA
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.isna().sum())

x = train_csv.drop(
    ['casual', 'registered', 'count'],
    axis=1
)

y = train_csv['count']

print("x shape :", x.shape)
print("y shape :", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=48
)


# 2. MODEL
model = Sequential()
model.add(Dense(512, input_dim=8))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))


# 3. COMPILE / TRAIN
model.compile(loss='mse', optimizer='adam')

start_time = time.time()

batch_size = 3

history = model.fit(
    x_train,
    y_train,
    epochs=358,
    batch_size=batch_size
)

train_time = time.time() - start_time


# 4. EVALUATE
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("loss :", loss)
print("R2 :", r2)
print("RMSE :", rmse)


# 5. KAGGLE TEST PREDICTION
y_submit = model.predict(test_csv).flatten()

submission['count'] = y_submit

submission.to_csv(
    path + "submit/submit_0904_1630.csv",
    index=True
)


# 6. MODEL HISTORY
my_util.record_model_csv(
    model=model,
    data_shape=x_train.shape,
    random_num=48,
    batch_size=batch_size,
    history=history,
    training_time=train_time,
    test_loss=loss,
    r2=r2
)
```

---
relu 
activation (활성화함수) → 레이어 층에 있는 활성화 함

y=relu(wx+b) → 0이하를 0으로 만드는 함수