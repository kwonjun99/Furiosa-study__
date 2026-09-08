# Day 07 - Binary Classification, Sigmoid, Binary Crossentropy

## 1. 오늘 학습 핵심

Day 07에서는 회귀 기본을 마무리하고 **이진분류(Binary Classification)** 로 넘어갔다.

```text
데이터 준비
↓
Train / Test 분리
↓
stratify로 클래스 비율 유지
↓
Dense 모델 구성
↓
마지막 층 Sigmoid
↓
Binary Crossentropy
↓
Accuracy 평가
↓
0.5 threshold로 0/1 변환
↓
Kaggle 제출
```

---

## 2. EarlyStopping 복습

EarlyStopping은 최대 epoch를 크게 잡고, Validation 성능이 더 이상 좋아지지 않을 때 자동으로 학습을 멈추는 방식으로 많이 사용한다.

```python
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)
```

핵심:

```text
epochs
= 최대 학습 가능 횟수

patience
= val_loss가 좋아지지 않아도 기다리는 epoch 수

restore_best_weights=True
= 가장 좋은 val_loss 시점의 weight 복원
```

---

## 3. Local Minimum / Global Minimum

신경망의 loss landscape에서 **진짜 global minimum을 반드시 찾는 것은 일반적으로 어렵다.**

결과는 다음에 따라 달라질 수 있다.

```text
초기 weight
optimizer
learning rate
batch size
모델 구조
```

실무에서는 global minimum을 증명해서 찾기보다:

> **Validation/Test 성능이 좋은 충분히 좋은 minimum을 찾는 것**

이 더 중요하다.

---

## 4. 회귀와 이진분류 차이

### 회귀 Regression

연속적인 숫자를 예측한다.

```text
집값
온도
자전거 대여량
```

마지막 층:

```python
Dense(1)
```

대표 loss:

```python
loss='mse'
```

대표 평가:

```text
RMSE
MAE
R²
```

### 이진분류 Binary Classification

두 클래스 중 하나를 예측한다.

```text
0 / 1
정상 / 암
합격 / 불합격
구매 / 미구매
```

마지막 층:

```python
Dense(1, activation='sigmoid')
```

대표 loss:

```python
loss='binary_crossentropy'
```

대표 평가:

```text
accuracy
```

---

## 5. sklearn Dataset / Bunch

scikit-learn의 많은 dataset 객체는 `Bunch` 형태라서 두 방식 모두 가능하다.

```python
x = datasets['data']
```

또는:

```python
x = datasets.data
```

의미:

```text
datasets['data']
→ dictionary처럼 접근

datasets.data
→ object attribute처럼 접근
```

실무 데이터에서는 CSV를 많이 사용하므로 Pandas가 중요하다.

```python
pd.read_csv()
```

---

## 6. stratify

분류 문제에서는 train/test 분리 시 클래스 비율을 유지하기 위해 `stratify=y`를 자주 사용한다.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y
)
```

중요:

> `stratify=y`는 0과 1의 개수를 50:50으로 만드는 것이 아니다.

예를 들어 전체 데이터가:

```text
0 : 90%
1 : 10%
```

이라면 Train/Test에서도 대략 같은 비율을 유지한다.

```text
Train
0 : 90%
1 : 10%

Test
0 : 90%
1 : 10%
```

확인:

```python
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))
```

---

## 7. 데이터 불균형

분류 데이터에서 한 클래스가 지나치게 많을 수 있다.

```text
0 : 99%
1 : 1%
```

이 경우 단순 accuracy만 보면 착시가 생길 수 있다.

다만:

```text
Train 데이터는 무조건 50:50이어야 한다
```

는 것은 아니다.

향후 배우게 될 수 있는 방법:

```text
Oversampling
Undersampling
Class Weight
```

---

## 8. Sigmoid

이진분류의 마지막 층에서 많이 사용하는 활성화 함수.

```python
model.add(
    Dense(1, activation='sigmoid')
)
```

Sigmoid 함수:

\[
\sigma(x)=rac{1}{1+e^{-x}}
\]

특징:

```text
입력값
-∞ ~ +∞

↓

출력값
0 ~ 1
```

예:

```text
0.13
0.48
0.73
0.96
```

이진분류에서는 이 값을 class 1 쪽의 확률처럼 해석할 수 있다.

---

## 9. 0.5 Threshold

Sigmoid 출력은 바로 0/1이 아니라 연속적인 값이다.

```text
0.13
0.92
0.67
0.21
```

최종 class를 결정할 때는 보통:

```text
0.5 미만 → 0
0.5 이상 → 1
```

로 나눈다.

코드:

```python
y_predict_class = (y_predict >= 0.5).astype(int)
```

또는 현재 수업에서는:

```python
y_predict_class = np.round(y_predict)
```

처럼 처리할 수도 있다.

---

## 10. 왜 Step Function이 아니라 Sigmoid인가?

Step Function은 바로 0/1을 출력할 수 있지만 대부분 구간에서 미분값이 0이라 학습에 적합하지 않다.

신경망 학습 흐름:

```text
Loss
↓
미분
↓
Gradient
↓
Backpropagation
↓
Weight 수정
```

Sigmoid는 부드러운 함수이기 때문에 미분할 수 있다.

따라서 학습 중에는:

```text
0.137
0.482
0.731
0.967
```

같은 연속값을 사용하고,

학습 후:

```text
threshold
↓
0 / 1
```

로 최종 class를 결정한다.

---

## 11. Gradient / Weight Update

Gradient:

> **weight를 조금 움직였을 때 loss가 어느 방향으로 얼마나 변하는지 알려주는 값**

기본적인 weight update:

\[
w_{new}=w-\eta 	imes gradient
\]

여기서:

```text
Loss
= 얼마나 틀렸는지

Gradient
= 어느 방향으로 수정해야 하는지

Learning Rate
= 얼마나 크게 수정할지

Optimizer
= 실제 weight 업데이트 방법
```

현재 Keras에서는:

```python
optimizer='adam'
```

이 이 과정을 자동으로 처리한다.

---

## 12. Binary Crossentropy

이진분류 대표 loss 함수.

```python
loss='binary_crossentropy'
```

의미:

> **정답 0/1에 대해 모델이 얼마나 적절한 확률을 출력했는지 평가**

정답이 1일 때:

```text
예측 0.99
→ 잘 맞음
→ loss 작음

예측 0.01
→ 강하게 틀림
→ loss 매우 큼
```

정답이 0이면 반대다.

전체 흐름:

```text
Sigmoid
↓
0~1 출력
↓
Binary Crossentropy
↓
정답과 확률 비교
↓
Loss
↓
Backpropagation
↓
Weight Update
```

---

## 13. Binary Crossentropy에서 log를 쓰는 이유

확률 곱셈은 log를 쓰면 덧셈으로 바뀐다.

```text
확률 곱셈
↓
log
↓
덧셈
```

또한 정답과 반대로 강하게 확신한 예측에 큰 loss를 줄 수 있다.

예:

```text
정답 = 1

예측 0.9
→ 작은 loss

예측 0.1
→ 큰 loss

예측 0.01
→ 매우 큰 loss
```

---

## 14. Accuracy

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
```

Accuracy:

```text
맞춘 개수 / 전체 개수
```

예:

```text
100개 중 95개 정답
→ accuracy = 0.95
```

---

## 15. Classification metrics 오류

다음과 같은 오류가 날 수 있다.

```text
Classification metrics can't handle
a mix of binary and continuous targets
```

이유:

```text
y_test
= 0/1 class

y_predict
= sigmoid의 연속적인 출력
```

예:

```text
y_test
[0, 1, 1, 0]

y_predict
[0.12, 0.83, 0.64, 0.27]
```

따라서 accuracy_score를 사용하기 전에는 class로 변환해야 한다.

```python
y_predict_class = np.round(y_predict)
```

또는:

```python
y_predict_class = (y_predict >= 0.5).astype(int)
```

그 다음:

```python
acc_score = accuracy_score(
    y_test,
    y_predict_class
)
```

---

## 16. 회귀 vs 이진분류 정리

| 구분 | 회귀 Regression | 이진분류 Binary Classification |
|---|---|---|
| 목표 | 연속적인 숫자 예측 | 0/1 예측 |
| 예 | 집값, 대여량 | 정상/암 |
| y 형태 | 실수 | 0 또는 1 |
| 마지막 층 | `Dense(1)` | `Dense(1, activation='sigmoid')` |
| 출력 | 자유로운 실수 | 0~1 |
| 대표 loss | MSE, MAE | Binary Crossentropy |
| 대표 평가 | RMSE, MAE, R² | Accuracy |
| Threshold | 사용 안 함 | 사용 |
| 예측 후 0/1 변환 | X | O |

```text
회귀
Dense(1)
↓
MSE
↓
RMSE / R²


이진분류
Dense(1, sigmoid)
↓
Binary Crossentropy
↓
Threshold
↓
Accuracy
```

---

## 17. Santander 이진분류 모델

```python
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import time
import pandas as pd
import my_util

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. DATA
path = "./_data/kaggle_santander/"

train_csv = pd.read_csv(
    path + "train.csv",
    index_col=0
)

test_csv = pd.read_csv(
    path + "test.csv",
    index_col=0
)

submission = pd.read_csv(
    path + "sample_submission.csv",
    index_col=0
)

x = train_csv.drop(
    ['target'],
    axis=1
)

y = train_csv['target']

print("x shape :", x.shape)
print("y shape :", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=78,
    stratify=y
)

print(np.unique(
    y_train,
    return_counts=True
))

print(np.unique(
    y_test,
    return_counts=True
))


# 2. MODEL
model = Sequential()

model.add(
    Dense(
        60,
        input_dim=200,
        activation='relu'
    )
)

model.add(
    Dense(
        50,
        activation='relu'
    )
)

model.add(
    Dense(
        40,
        activation='relu'
    )
)

model.add(
    Dense(
        30,
        activation='relu'
    )
)

model.add(
    Dense(
        20,
        activation='relu'
    )
)

model.add(
    Dense(
        10,
        activation='relu'
    )
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# 3. COMPILE / TRAIN
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

start_time = time.time()

batch_size = 700

history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)

train_time = time.time() - start_time


# 4. EVALUATE
loss = model.evaluate(
    x_test,
    y_test
)

print("========================")
print("loss :", round(loss[0], 4))
print("acc  :", round(loss[1], 4))


# 5. PREDICT
y_predict = model.predict(
    x_test
)

y_predict_class = (
    y_predict >= 0.5
).astype(int)

acc_score = accuracy_score(
    y_test,
    y_predict_class
)

print("accuracy_score :", acc_score)


# 6. KAGGLE SUBMISSION
y_submit = model.predict(
    test_csv
).flatten()

submission['target'] = y_submit

submission.to_csv(
    path + "submit/submit_0908_1630.csv",
    index=True
)
```

---

