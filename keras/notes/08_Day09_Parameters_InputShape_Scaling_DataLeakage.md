# Day 09 - Model Parameters / Input Shape / Scaling / Data Leakage

## 1. 오늘 학습 핵심

Day 09에서는 지금까지 배운 분류 모델을 확장하면서 다음 내용을 정리했다.

```text
Softmax 복습
↓
Pandas Series / NumPy reshape
↓
model.summary()
↓
Dense Parameter 계산
↓
Loss vs Metric
↓
ROC-AUC
↓
input_shape
↓
고차원 입력 데이터
↓
데이터 전처리
↓
MinMax Scaling
↓
Data Leakage
↓
기존 모델에 Scaling 적용 후 성능 비교
```

---

## 2. Santander 이진분류 → Softmax 형태 실험

Santander의 target은 원래 `0 / 1`인 이진분류 데이터다.

일반적인 방식:

```python
model.add(Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

학습 실험 차원에서 One-Hot Encoding 후:

```text
0 → [1, 0]
1 → [0, 1]
```

로 만들고,

```python
model.add(Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

형태로 구성할 수도 있다.

단, Softmax 방식이 자동으로 더 좋은 것은 아니다. 같은 validation/test 데이터에서 Accuracy, ROC-AUC 등 동일한 metric으로 비교해야 한다.

---

## 3. Pandas Series와 reshape

CSV에서:

```python
y = train_csv['target']
```

를 하면:

```python
print(type(y))
# <class 'pandas.core.series.Series'>
```

이다.

Pandas Series에는 NumPy의 `reshape()`가 없기 때문에:

```python
y.reshape(-1, 1)
```

을 바로 사용하면 오류가 발생한다.

따라서:

```text
train_csv['target']
↓
Pandas Series
↓
.to_numpy()
↓
NumPy Array
↓
.reshape(-1, 1)
↓
OneHotEncoder
```

순서로 변환한다.

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

y = train_csv['target'].to_numpy()
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
```

한 줄로도 가능하다.

```python
y = train_csv['target'].to_numpy().reshape(-1, 1)
y = ohe.fit_transform(y)
```

---

## 4. model.summary()

예제:

```python
model = Sequential()

model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
```

`model.summary()`를 사용하면:

```text
Layer 이름
Output Shape
Parameter 개수
전체 Parameter
Trainable Parameter
```

등을 확인할 수 있다.

---

## 5. Dense Parameter 계산

Dense layer의 Parameter 계산 공식:

```text
Parameter
= Weight 개수 + Bias 개수
```

수식:

\[
Params
=
(입력 노드 수 × 출력 노드 수)
+
출력 노드 수
\]

즉:

```text
Weight
= 입력 노드 × 출력 노드

Bias
= 출력 노드 수
```

Bias는 연결마다 존재하는 것이 아니라 각 출력 뉴런마다 하나씩 존재한다.

예제 모델:

```python
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
```

| Layer | Weight | Bias | Params |
|---|---:|---:|---:|
| 1 → 3 | 1×3 = 3 | 3 | 6 |
| 3 → 4 | 3×4 = 12 | 4 | 16 |
| 4 → 3 | 4×3 = 12 | 3 | 15 |
| 3 → 1 | 3×1 = 3 | 1 | 4 |

총 Parameter:

```text
6 + 16 + 15 + 4 = 41
```

---

## 6. Dense 내부 계산

뉴런 하나의 기본 계산은:

\[
z=w^Tx+b
\]

여러 샘플을 행렬 형태로 처리하는 Keras Dense에서는 보통:

\[
Z=XW+b
\]

형태로 생각하면 된다.

Activation이 있다면:

\[
Y=activation(XW+b)
\]

예:

```python
Dense(4, activation='relu')
```

내부 흐름:

```text
X
↓
XW + b
↓
ReLU
↓
Output
```

엄밀한 수학 용어로는:

```text
Wx
= Linear Transformation

Wx + b
= Affine Transformation
```

이다.

---

## 7. Loss vs Metric

### Loss

```text
학습 중 Weight를 어떻게 수정할지 결정하는 기준
```

예:

```text
MSE
Binary Crossentropy
Categorical Crossentropy
```

흐름:

```text
Prediction
↓
Loss
↓
Gradient
↓
Backpropagation
↓
Weight Update
```

### Metric

```text
학습된 모델이 실제로 얼마나 잘하고 있는지 평가하는 기준
```

예:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
RMSE
R²
```

핵심:

```text
Loss
↓
학습 방향 결정

Metric
↓
모델 성능 평가

Competition Metric
↓
최종적으로 어떤 모델을 좋은 모델이라고 할지 결정
```

서로 다른 Loss의 숫자 크기를 직접 비교해서 어느 모델이 더 좋다고 판단하면 안 된다.

---

## 8. ROC-AUC

Accuracy는:

```text
정답을 몇 개 맞혔는가?
```

를 본다.

하지만 불균형 데이터에서는 Accuracy만으로 성능을 판단하면 문제가 생길 수 있다.

예:

```text
class 0 = 90%
class 1 = 10%
```

모든 데이터를 0이라고 예측해도 Accuracy가 90%가 될 수 있다.

ROC-AUC는 단순히 0/1을 몇 개 맞혔는지만 보는 것이 아니라,

> 양성 샘플에 음성 샘플보다 더 높은 예측 점수를 얼마나 잘 부여하는가

를 평가하는 metric으로 이해하면 된다.

Santander처럼 불균형한 이진분류에서는 Accuracy만 보는 것보다 ROC-AUC가 더 중요할 수 있다.

---

## 9. input_shape

기존에는 주로 2차원 tabular 데이터를 사용했다.

예:

```text
(200, 3)

200 samples
3 features
```

Keras의 `input_shape`에는 batch/sample 차원을 제외한 나머지 shape를 적는다.

```text
실제 데이터 Shape
↓
input_shape

(n, 4)
→ (4,)

(n, 100, 3)
→ (100, 3)

(n, 100, 100, 3)
→ (100, 100, 3)
```

예:

```python
model.add(
    Dense(
        60,
        input_shape=(4,),
        activation='relu'
    )
)
```

현대 Keras 스타일:

```python
from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(4,)),
    Dense(60, activation='relu')
])
```

---

## 10. 2차원 / 3차원 / 4차원 데이터

### 2차원

```text
(samples, features)

예:
(150, 4)
```

Iris 같은 tabular 데이터.

### 3차원

```text
(samples, timesteps, features)

예:
(1000, 30, 5)
```

시계열 / RNN 데이터에서 자주 사용한다.

Keras input_shape:

```python
input_shape=(30, 5)
```

### 4차원

이미지 batch 예:

```text
(samples, height, width, channels)

예:
(1000, 224, 224, 3)
```

Keras input_shape:

```python
input_shape=(224, 224, 3)
```

차원 수는 Keras나 PyTorch가 결정하는 것이 아니라 데이터 구조와 사용하는 layer 종류에 따라 결정된다.

---

## 11. 데이터 전처리

Data Preprocessing:

> 원본 데이터를 모델이 학습하기 좋은 형태로 정리하고 변환하는 과정

대표 과정:

```text
원본 데이터
↓
데이터 확인
↓
Train / Validation / Test 분리
↓
결측치 처리
↓
이상치 처리
↓
범주형 데이터 Encoding
↓
Scaling
↓
Feature 선택 / 생성
↓
모델 학습
```

주의:

전처리 중에서도 데이터에서 통계량을 학습하는 전처리(Scaling, Imputation 등)는 Train 데이터로만 fit 해야 한다.

---

## 12. Scaling

Scaling:

> Feature마다 서로 다른 값의 크기(scale)를 비슷한 수준으로 맞춰주는 전처리 방법

예:

```text
나이      25
키       175
연봉      50,000,000
```

Feature마다 숫자 크기가 너무 다르면 신경망의 Gradient 기반 학습이 비효율적일 수 있다.

Scaling을 하면:

```text
Feature scale 차이 감소
↓
Gradient 계산 안정화
↓
Optimizer가 학습하기 쉬워짐
↓
수렴 속도 / 성능 개선 가능
```

---

## 13. MinMaxScaler

MinMax Scaling 수식:

\[
x' = (x - xmin) / (xmax - xmin)
\]

Train 데이터 기준으로 일반적으로 최소값을 0, 최대값을 1로 변환한다.

예:

```text
원본
10, 15, 20

↓

MinMax

0, 0.5, 1
```

코드:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

---

## 14. x / xmax 와 MinMax 차이

단순히:

\[
x' = x / xmax
\]

를 사용할 수도 있다.

예:

```text
[10, 15, 20]

x / xmax

→ [0.5, 0.75, 1.0]
```

반면 MinMax:

```text
[10, 15, 20]
→ [0, 0.5, 1]
```

MinMax Scaling은 원점 기준의 비율은 달라질 수 있지만:

```text
데이터 순서 유지
상대적 간격 구조 유지
```

가 된다.

이는 데이터를 무작위로 왜곡하는 것이 아니라 전체 좌표계를 이동하고 일정한 비율로 확대/축소하는 Affine Transformation이다.

---

## 15. Scaling의 목적

Scaling의 핵심 목적은:

> 원래 숫자의 비율 자체를 보존하는 것이 아니라 Feature들 사이의 scale 차이를 줄여 모델이 학습하기 쉽게 만드는 것

이다.

특히:

```text
Neural Network
Logistic Regression
SVM
KNN
PCA
```

등에서 Scaling이 중요하다.

Tree 계열:

```text
Decision Tree
Random Forest
XGBoost
LightGBM
```

은 Scaling의 영향이 상대적으로 적다.

---

## 16. Data Leakage

잘못된 예:

```python
scaler = MinMaxScaler()

scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=49
)
```

문제는 `scaler.fit(x)` 하는 순간 전체 데이터의 xmin / xmax를 사용한다는 점이다.

즉 아직 평가에만 써야 할 Test 데이터 정보까지 Scaling 기준에 들어간다.

이를:

```text
Data Leakage
```

라고 한다.

---

## 17. 올바른 Scaling 순서

먼저 Train/Test를 나눈다.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=49
)
```

그 다음 Train 데이터로만 fit한다.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

핵심:

```text
Train
→ fit + transform

Validation
→ transform

Test
→ transform
```

`fit`:

```text
Train 데이터에서 xmin / xmax 등 Scaling 기준 학습
```

`transform`:

```text
학습한 기준으로 실제 데이터 변환
```

---

## 18. Test 값이 반드시 0~1인 것은 아니다

Train에서:

```text
xmin = 10
xmax = 40
```

을 학습했다고 하자.

Test에:

```text
100
```

이라는 값이 들어오면:

```text
(100 - 10) / (40 - 10)
= 3.0
```

이 될 수 있다.

즉 MinMaxScaler를 사용했다고 해서 Test 데이터가 반드시 0~1 범위에 들어가는 것은 아니다.

---


