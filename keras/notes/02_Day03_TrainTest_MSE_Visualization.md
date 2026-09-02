# Day 03 - Train/Test Split, MSE, Matplotlib, 실무 데이터셋

## 1. 신경망 기본식 복습

신경망의 가장 기본적인 연산은 다음과 같이 생각할 수 있다.

\[
y = wx + b
\]

- `x` : 입력 데이터(input), feature
- `w` : 가중치(weight)
- `b` : 편향(bias)
- `y` : 모델의 출력값(output, prediction)

feature가 여러 개라면:

\[
y = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + b
\]

신경망은 이런 `wx+b` 연산을 여러 층에 걸쳐 반복한다.

```text
입력 x
  ↓
wx + b
  ↓
activation
  ↓
다음 layer
  ↓
prediction
```

학습 과정:

```text
Weight
  ↓
예측
  ↓
Loss 계산
  ↓
Optimizer
  ↓
Weight / Bias 수정
  ↓
다시 예측
```

- `loss` : 예측값과 실제값의 차이를 수치화
- `optimizer` : loss가 작아지도록 weight와 bias를 수정
- `training` : 좋은 weight와 bias를 찾는 과정
- `inference` : 학습된 weight와 bias로 새로운 입력의 결과를 예측하는 과정

---

## 2. Epoch의 의미

```python
model.fit(x_train, y_train, epochs=100)
```

`epoch=100`은 **전체 학습 데이터를 100번 반복해서 학습**한다는 뜻이다.

epoch가 많다고 무조건 성능이 좋아지는 것은 아니다. 너무 많이 학습하면 Overfitting(과적합)이 발생할 수 있다.

---

## 3. 데이터 슬라이싱

```python
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = x[:7]
y_train = y[:7]

x_test = x[7:]
y_test = y[7:]
```

이렇게 순서대로 자르면 Train과 Test의 분포가 달라질 수 있다.

```text
Train : 1 2 3 4 5 6 7
Test  : 8 9 10
```

일반적인 독립 데이터에서는 shuffle 후 분할하는 것이 보통 더 적절하다.

> 단, 시계열처럼 시간 순서가 중요한 데이터는 무작정 shuffle하면 안 된다.

---

## 4. train_test_split

설치:

```bash
pip install scikit-learn
```

사용:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    test_size=0.3,
    shuffle=True,
    random_state=294
)
```

- `train_size=0.7` : 전체의 70%를 학습 데이터로 사용
- `test_size=0.3` : 전체의 30%를 테스트 데이터로 사용
- `shuffle=True` : 데이터를 섞은 후 분리
- `random_state` : 랜덤 분할 결과를 재현하기 위한 seed

`train_size`, `test_size`를 둘 다 생략하면 일반적으로 `test_size=0.25`가 사용된다.

### random_state

숫자 자체에 특별한 의미는 없다.

- 같은 숫자 → 같은 분할 결과 재현
- 다른 숫자 → 다른 분할 결과가 나올 수 있음

> 테스트 성능이 좋아지는 `random_state`를 찾아 계속 바꾸는 것은 좋은 모델 개선 방법이 아니다. seed는 실험 재현을 위해 고정하는 것이 좋다.

---

## 5. Train/Test를 나누는 이유

```text
Train
→ 모델의 Weight를 학습하는 데이터

Test
→ 학습이 끝난 모델이 처음 보는 데이터에서 잘 동작하는지 확인
```

같은 데이터로 학습과 평가를 모두 하면 모델이 이미 본 데이터에서 얼마나 잘 맞추는지만 확인하게 된다.

우리가 알고 싶은 것은 **처음 보는 데이터에서도 잘 예측하는가?** 이다.

이 능력을 `Generalization(일반화)`이라고 한다.

---

## 6. 데이터 Shape과 Dense

```text
x.shape = (20, 1)
           │   │
           │   └─ feature 1개
           └──── sample 20개
```

```python
model.add(Dense(20, input_dim=1))
```

- `input_dim=1` : 입력 feature가 1개
- `Dense(20)` : 해당 layer의 neuron 수가 20개

즉 데이터가 20개라고 해서 `Dense(20)`을 사용해야 하는 것은 아니다.

---

## 7. Dense Layer의 Parameter 수

\[
Parameters = 입력노드수 \times 출력노드수 + 출력노드수
\]

마지막 `+ 출력노드수`는 bias 개수이다.

예:

```text
입력 8개 → Dense(10)
```

\[
8 \times 10 + 10 = 90
\]

Parameter가 많아질수록:

- 연산량 증가
- 메모리 사용 증가
- 학습 시간 증가
- 추론 latency 증가 가능
- 과적합 위험 증가

Dense 숫자가 크고 layer가 많다고 무조건 좋은 모델은 아니다.

---

## 8. Matplotlib

설치:

```bash
pip install matplotlib
```

Import:

```python
import matplotlib.pyplot as plt
```

Matplotlib은 AI에서 **데이터와 모델을 분석하고 진단하기 위한 시각화 도구**로 많이 사용한다.

주요 용도:

1. 데이터 분포 확인
2. Train/Test 데이터 비교
3. Loss 변화 확인
4. 실제값과 예측값 비교
5. Overfitting 확인

기본 사용법:

```python
plt.plot(x, y)
plt.scatter(x, y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('graph')
plt.legend()
plt.grid()
plt.show()
```

Train/Test 분포 확인:

```python
plt.scatter(x_train, y_train, label='train')
plt.scatter(x_test, y_test, label='test')
plt.legend()
plt.show()
```

모델 예측값 시각화:

```python
results = model.predict(x)

plt.scatter(x, y, label='real')
plt.plot(x, results, label='prediction')
plt.legend()
plt.show()
```

---

## 9. Loss가 0이면 무조건 Overfitting인가?

아니다.

Loss가 0에 가깝다는 것은 **현재 평가한 데이터에서 예측 오차가 매우 작다**는 뜻이다.

Overfitting은 보통:

```text
Train Loss는 매우 낮음
하지만
Validation/Test Loss는 높음
```

처럼 **학습 데이터와 보지 않은 데이터의 성능 차이**를 보고 판단한다.

따라서 `loss = 0` 자체만으로 과적합이라고 판단하면 안 된다.

---

## 10. MSE

MSE = **Mean Squared Error, 평균제곱오차**

\[
MSE = \frac{1}{n}\sum(y-\hat y)^2
\]

- `y` : 실제값
- `ŷ` : 예측값

계산 과정:

```text
실제값 - 예측값
       ↓
      제곱
       ↓
 전체 데이터 평균
```

Keras:

```python
model.compile(
    loss='mse',
    optimizer='adam'
)
```

의 의미:

> 예측값과 실제값 사이의 평균제곱오차를 loss로 사용하고, Adam optimizer로 MSE가 작아지도록 weight와 bias를 수정한다.

---

## 11. Optimizer와 Chain Rule

Backpropagation에서는 미분과 Chain Rule(연쇄법칙)을 이용해 각 weight가 loss에 얼마나 영향을 주는지 계산한다.

```text
Loss
 ↓
Gradient 계산
 ↓
Backpropagation
 ↓
Optimizer
 ↓
Weight Update
```

현재 단계에서는 **미분을 이용해 Loss가 작아지는 방향으로 Weight를 갱신한다** 정도로 이해한다.

---

## 12. California Housing

```python
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
```

새로운 데이터셋을 받으면 항상 `shape`를 먼저 확인한다.

California Housing은 feature가 8개이므로 첫 layer의 입력 차원은 8이다.

```python
model.add(Dense(10, input_dim=8))
```

예제:

```python
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=49
)

model = Sequential()
model.add(Dense(1, input_dim=8))
model.add(Dense(40))
model.add(Dense(89))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(79))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=385, batch_size=30)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)
```

평가와 예측의 차이:

```python
model.evaluate(x_test, y_test)
```

→ Test 전체의 loss를 계산

```python
results = model.predict(x_test)
```

→ 각 sample의 실제 예측값을 출력

```python
results = model.predict(x_test)

print("예측값 :", results[:10].flatten())
print("실제값 :", y_test[:10])
```

`results[:10].flatten()`:

- `[:10]` : 앞의 10개만 선택
- `.flatten()` : `(10, 1)` 같은 배열을 `(10,)` 형태의 1차원 배열로 펼침

---

## 13. Boston Housing 예제

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

# 1. Data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. Model
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(30))
model.add(Dense(62))
model.add(Dense(48))
model.add(Dense(23))
model.add(Dense(1))

# 3. Compile / Train
model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=34
)

# 4. Evaluate
loss = model.evaluate(x_test, y_test)

print('loss :', loss)
```

`input_dim=13`인 이유는 한 sample의 feature가 13개이기 때문이다.

---

## 14. LLM / VLM / VLA

| 모델 | 입력 | 출력 | 핵심 |
|---|---|---|---|
| LLM | Text | Text | 언어 이해 / 생성 |
| VLM | Image/Video + Text | Text | 보고 이해 |
| VLA | Image/Video + Text/Goal | Action | 보고 이해하고 행동 |

모빌리티 예:

```text
"집 가는 길에 충전소 들러줘."

LLM / Agent
→ 사용자 의도 이해
→ 충전소 검색
→ 경로 계획

VLM
→ 주변 도로 상황 이해

VLA
→ 실제 주행 행동
```

간단히 기억하면:

```text
LLM → 말 이해
VLM → 세상 보기
VLA → 보고 이해하고 행동
```

---

## 15. Day 03 핵심 정리

1. `y = wx + b`는 신경망의 기본 연산이다.
2. Training은 Loss가 작아지는 Weight와 Bias를 찾는 과정이다.
3. Epoch는 전체 Train 데이터를 반복 학습하는 횟수다.
4. Train/Test를 분리하는 이유는 처음 보는 데이터에서의 성능을 확인하기 위해서다.
5. random_state는 데이터 분할을 재현하기 위한 값이다.
6. 데이터의 sample 수와 Dense neuron 수는 관계가 없다.
7. input_dim은 feature 개수와 맞아야 한다.
8. Dense가 커질수록 Parameter와 연산량이 증가한다.
9. Matplotlib은 데이터와 모델을 시각적으로 진단하는 도구다.
10. MSE는 실제값과 예측값의 차이를 제곱해 평균낸 Loss 함수다.
11. Backpropagation에서는 미분과 Chain Rule을 이용해 Weight를 갱신한다.
12. 좋은 모델은 학습 데이터를 단순히 외우는 것이 아니라 새로운 데이터에서도 잘 동작해야 한다.

## 한 줄 요약

> **Day 03에서는 데이터를 Train/Test로 올바르게 분리하고, Dense 모델을 학습한 뒤 MSE와 Matplotlib을 이용해 모델의 일반화 성능을 평가하는 기본 흐름을 학습했다.**
