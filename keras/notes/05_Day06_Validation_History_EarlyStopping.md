# Day 06 - Validation, History, Loss Graph, EarlyStopping

## 1. 오늘 학습 핵심

Day 06에서는 모델을 단순히 학습시키는 것에서 한 단계 더 나아가 다음 내용을 학습했다.

- `verbose`
- Train / Validation / Test 구분
- `loss` / `val_loss`
- `history.history`
- 학습 시간 측정
- Matplotlib loss graph
- EarlyStopping
- 최적 weight 복원

전체 흐름:

```text
데이터 준비
↓
Train / Validation / Test 구분
↓
model.fit()
↓
loss / val_loss 기록
↓
그래프로 학습 상태 확인
↓
과적합 여부 판단
↓
EarlyStopping으로 학습 종료
↓
최적 weight 복원
↓
Test 데이터로 최종 평가
```

---

## 2. verbose

`verbose`는 학습 진행 상황을 화면에 얼마나 자세히 출력할지 정하는 옵션이다.

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=30,
    verbose=1
)
```

```text
verbose=0
→ 출력하지 않음

verbose=1
→ progress bar 형태로 출력

verbose=2
→ progress bar 없이 epoch 단위로 출력
```

현재는 `0`, `1`, `2` 정도만 기억하면 충분하다.

---

## 3. Train / Validation / Test

```text
Train      = 공부
Validation = 중간 모의고사
Test       = 최종 시험
```

### Train
모델이 실제로 weight를 업데이트하는 데이터.

```text
forward
↓
loss 계산
↓
backpropagation
↓
weight update
```

### Validation
학습 중간에 현재 모델이 처음 보는 데이터에서도 잘 작동하는지 확인하는 데이터.

주요 역할:

```text
과적합 발견
모델 선택
epoch 선택
hyperparameter 비교
EarlyStopping 기준
```

### Test
모든 학습과 모델 선택이 끝난 뒤 최종 성능을 확인하는 데이터.

```python
loss = model.evaluate(x_test, y_test)
```

---

## 4. 데이터를 직접 8 / 4 / 4로 나누기

```python
import numpy as np
from sklearn.model_selection import train_test_split

x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    shuffle=True,
    random_state=34
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.67,
    shuffle=True,
    random_state=34
)
```

대략:

```text
Train      = 8개
Validation = 4개
Test       = 4개
```

---

## 5. validation_split

직접 `x_val`, `y_val`을 만들지 않고 `fit()`에서 간단히 validation 데이터를 만들 수 있다.

```python
history = model.fit(
    x_train,
    y_train,
    epochs=345,
    batch_size=30,
    verbose=1,
    validation_split=0.2
)
```

의미:

```text
현재 x_train / y_train 중
20%를 validation 용도로 사용
```

간단한 실험에서는 `validation_split`이 편하고,
데이터 구성을 직접 제어해야 할 때는 `x_val`, `y_val`을 직접 나누는 방법이 좋다.

---

## 6. loss와 val_loss

### loss
Train 데이터에서의 오차.

### val_loss
Validation 데이터에서의 오차.

모델의 일반화 성능과 과적합 여부를 볼 때는 `loss`만 보는 것보다 `val_loss`가 중요하다.

예:

```text
Epoch 1
loss     = 100
val_loss = 110

Epoch 50
loss     = 30
val_loss = 35

Epoch 100
loss     = 10
val_loss = 60
```

Train loss는 계속 내려가지만 Validation loss가 다시 올라가면 과적합 가능성이 있다.

핵심:

> `loss`는 계속 내려가도 `val_loss`가 올라가기 시작하면 과적합 신호가 될 수 있다.

---

## 7. Validation과 Evaluation 차이

```text
Validation
= 학습 중간중간 모델 상태 확인

Evaluation
= 학습 완료 후 최종 Test 성능 확인
```

예:

```python
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2
)

loss = model.evaluate(
    x_test,
    y_test
)
```

---

## 8. 학습 시간 측정

```python
import time

start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=30
)

train_time = time.time() - start_time

print(round(train_time, 2))
```

`round(값, 자리수)`는 반올림에 사용한다.

예:

```python
round(12.34567, 2)
```

결과:

```text
12.35
```

---

## 9. model.fit()과 history

```python
history = model.fit(...)
```

`model.fit()`은 학습을 수행하고 각 epoch의 기록을 `History` 객체로 반환한다.

```python
print(history.history)
```

예:

```python
{
    'loss': [1203.2, 13.9, 9.3, 44.4, 19.4, 6.4, 1.2, 26.0, 1.6, 1.0],
    'val_loss': [3.5, 1.3, 2.5, 1.2, 1.2, 1.2, 1.7, 1.2, 1.2, 1.2]
}
```

`history.history`는 Dictionary이고,

```python
history.history['loss']
history.history['val_loss']
```

처럼 key를 이용해 각 List를 꺼낼 수 있다.

Epoch가 10번이면 loss와 val_loss 값도 각각 10개가 저장된다.

---

## 10. Loss Graph

```python
import matplotlib.pyplot as plt

print(history.history['loss'])
print(history.history['val_loss'])

plt.figure(figsize=(9, 6))

plt.plot(
    history.history['loss'][3:],
    c='red',
    label='loss'
)

plt.plot(
    history.history['val_loss'][3:],
    c='blue',
    label='val_loss'
)

plt.legend(loc='upper right')
plt.title('California Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
```

### figsize

```python
plt.figure(figsize=(9, 6))
```

그래프 전체 그림판 크기.

```text
9 = 가로
6 = 세로
```

단위는 inch.

### `[3:]`

```python
history.history['loss'][3:]
```

앞의 3개 값을 제외하고 이후 값부터 그래프에 표시한다.

초반 loss가 너무 커서 그래프가 보기 어려울 때 사용할 수 있다.

### legend

```python
plt.legend(loc='upper right')
```

그래프 label 위치를 설정한다.

### grid

```python
plt.grid()
```

격자를 표시한다.

---

## 11. 한글 폰트

Windows에서 Matplotlib 한글이 깨질 경우:

```python
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

---

## 12. 최소 loss 찾기

```python
loss_list = history.history['loss']

min_loss = min(loss_list)
min_epoch = loss_list.index(min_loss) + 1

print("최소 loss :", min_loss)
print("최소 loss epoch :", min_epoch)
```

Validation 기준:

```python
val_loss_list = history.history['val_loss']

min_val_loss = min(val_loss_list)
min_val_epoch = val_loss_list.index(min_val_loss) + 1

print("최소 val_loss :", min_val_loss)
print("최적 epoch :", min_val_epoch)
```

---

## 13. EarlyStopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True
)
```

`model.fit()`에 callback으로 넣는다.

```python
history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=30,
    validation_split=0.2,
    callbacks=[early_stopping]
)
```

### monitor

```python
monitor='val_loss'
```

EarlyStopping이 어떤 값을 기준으로 볼지 설정한다.

### mode

```python
mode='auto'
```

감시하는 값이 커져야 좋은지, 작아져야 좋은지 자동으로 판단한다.

```text
loss / val_loss
→ 작을수록 좋음
→ min

accuracy
→ 클수록 좋음
→ max
```

`val_loss`라면 명시적으로:

```python
mode='min'
```

이라고 써도 된다.

### patience

```python
patience=10
```

Validation loss가 좋아지지 않더라도 바로 멈추지 않고 10 epoch 동안 기다린다.

### restore_best_weights

```python
restore_best_weights=True
```

학습이 멈춘 마지막 epoch의 weight가 아니라,
**가장 좋은 val_loss를 기록했던 epoch의 weight로 복원한다.**

예:

```text
Epoch 30
val_loss = 10   ← 가장 좋음

Epoch 31
11

...

Epoch 40
15   ← 학습 종료
```

`restore_best_weights=True`라면 최종 모델은 Epoch 30의 weight로 돌아간다.

---

## 14. Weight

기본 뉴런:

```text
y = wx + b
```

에서:

```text
w = weight
b = bias
```

모델은 학습하면서 weight와 bias를 계속 수정한다.

핵심:

```text
EarlyStopping
= 과적합 전에 학습 중지

restore_best_weights=True
= 가장 좋은 epoch의 weight 복원
```

---

## 15. 이름 표기

```python
EarlyStopping
```

은 **PascalCase**.

```python
restore_best_weights
validation_split
batch_size
```

는 **snake_case**.

---

## 16. Day 06 종합 예제

```python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split


# 1. DATA
x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=34
)


# 2. MODEL
model = Sequential()

model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. EARLY STOPPING
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)


# 4. COMPILE
model.compile(
    loss='mse',
    optimizer='adam'
)


# 5. TRAIN
history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=30,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)


# 6. BEST VAL LOSS
val_loss_list = history.history['val_loss']

min_val_loss = min(val_loss_list)
min_val_epoch = val_loss_list.index(min_val_loss) + 1

print("최소 val_loss :", min_val_loss)
print("최적 epoch :", min_val_epoch)


# 7. TEST EVALUATION
loss = model.evaluate(x_test, y_test)

print("Test loss :", loss)


# 8. GRAPH
plt.figure(figsize=(9, 6))

plt.plot(
    history.history['loss'],
    label='loss'
)

plt.plot(
    history.history['val_loss'],
    label='val_loss'
)

plt.legend(loc='upper right')
plt.title('Loss Graph')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()
```

--