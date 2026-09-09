# Day 08 - Multi-class Classification / Softmax / One-Hot Encoding


## 1. 오늘 학습 위치
Day 08에서는 **회귀 → 이진분류 → 다중분류**까지 확장했다.

```text
ANN / Neural Network
│
├─ MLP / Dense Network
│   └─ DNN
├─ CNN
│   ├─ ResNet
│   ├─ EfficientNet
│   └─ U-Net
├─ RNN
│   ├─ LSTM
│   └─ GRU
└─ Transformer
    ├─ BERT
    ├─ GPT
    ├─ ViT
    └─ LLM / VLM
```

- **ANN / NN**: 인공신경망 전체 범주
- **DNN**: 여러 층을 깊게 쌓은 신경망
- **CNN**: 이미지 처리
- **RNN / LSTM / GRU**: 순서·시계열 데이터
- **Transformer**: Attention 기반 구조, 현대 LLM의 핵심

현재 Dense 여러 층 구조는 **MLP 기반 DNN**이라고 볼 수 있다.

---

## 2. y값 종류에 따른 문제 유형

### 회귀
연속적인 숫자 예측

```text
집값
온도
자전거 대여량
```

### 이진분류
2개 클래스 중 하나 예측

```text
0 / 1
정상 / 암
합격 / 불합격
```

### 다중분류
3개 이상 클래스 중 하나 예측

```text
0 / 1 / 2
Iris 3종
숫자 0~9
```

---

처음 데이터를 받으면 다음을 확인하는 습관이 중요하다.

```text
데이터 개수
Feature 개수
Target 형태
Class 종류
Class 비율
결측치
최소 / 최대 / 평균 / 표준편차
Feature와 Target의 관계
```

Iris는 학습용으로 매우 잘 정제된 데이터다.

---

## 4. Train / Test Split + Stratify

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=333,
    shuffle=True,
    stratify=y,
)
```

`stratify=y`는 원래 클래스 비율을 Train/Test에 비슷하게 유지한다.

Iris는:

```text
0 : 50
1 : 50
2 : 50
```

이므로 분할 후에도 각 클래스 비율이 비슷하게 유지된다.

---

## 5. Softmax

다중분류에서는 출력 노드 수를 **클래스 개수**와 맞춘다.

Iris는 3개 클래스이므로:

```python
model.add(Dense(3, activation='softmax'))
```

Softmax 출력 예:

```text
[0.73, 0.22, 0.05]
```

핵심:

```text
각 값은 0~1 사이
전체 합 = 1
```

즉 클래스별 확률 분포처럼 해석할 수 있다.

```text
class 0 : 73%
class 1 : 22%
class 2 : 5%
```

가장 큰 값의 위치가 최종 예측 클래스가 된다.

> Softmax는 “모든 값을 1보다 작게 만드는 함수”보다  
> **각 출력값을 0~1 사이로 만들고 전체 합을 1로 만드는 함수**라고 이해하는 것이 정확하다.

---

## 6. One-Hot Encoding

원래 정수 label:

```text
0
1
2
```

을 위치 기반 벡터로 바꾼다.

```text
0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]
```

예:

```text
[0, 0, 1, 1, 2]

↓

[[1,0,0],
 [1,0,0],
 [0,1,0],
 [0,1,0],
 [0,0,1]]
```

Shape:

```text
(150,)
↓
(150, 3)
```

### One-Hot label

```text
[1,0,0]
[0,1,0]
[0,0,1]
```

이면:

```python
loss='categorical_crossentropy'
```

핵심:

```text
정수 Label
→ sparse_categorical_crossentropy

One-Hot Label
→ categorical_crossentropy
```

---

## 8. One-Hot Encoding 3가지 방법

### 1) Keras - to_categorical

```python
from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
```

가장 간단하다.

### 2) Pandas - get_dummies

```python
y = pd.get_dummies(y, dtype=int)
```

### 3) sklearn - OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
```

---

## 9. reshape()

원래:

```python
y.shape
# (150,)
```

은 1차원 배열이다.

`OneHotEncoder`는 입력을 보통:

```text
(samples, features)
```

형태의 2차원 배열로 받는다.

따라서:

```python
y = y.reshape(-1, 1)
```

로:

```text
(150,)
↓
(150, 1)
```

로 바꾼다.

의미:

```text
150 samples
1 categorical feature
```

### reshape 핵심 조건

`reshape()`는 **원소 개수와 순서는 유지하면서 shape만 바꾼다.**

예:

```python
a = np.array([1,2,3,4,5,6])
```

가능:

```python
a.reshape(6,1)
a.reshape(3,2)
a.reshape(2,3)
a.reshape(1,6)
```

불가능:

```python
a.reshape(4,2)
```

원래 원소는 6개인데 4×2는 8개가 필요하기 때문이다.

### `-1`의 의미

```python
y.reshape(-1, 1)
```

에서 `-1`은:

> 이 차원의 크기는 NumPy가 전체 원소 개수에 맞춰 자동으로 계산하라는 뜻

이다.

---

## 10. sparse_output=False

```python
ohe = OneHotEncoder(
    sparse_output=False
)
```

의 의미:

> One-Hot 결과를 희소행렬이 아니라 일반 NumPy 배열(Dense Array)로 반환한다.

One-Hot 데이터는 0이 많기 때문에 메모리를 아끼기 위해 희소행렬 형태를 사용할 수 있다.

현재 Keras 실습에서는 일반 배열로 확인하고 사용하기 편해서:

```python
sparse_output=False
```

를 사용한다.

※ `sparse`는 **희소행렬**이지 혼돈행렬이 아니다.

---

## 11. One-Hot Encoding의 단점

클래스 수가 많아지면 벡터가 매우 커진다.

```text
클래스 100,000개
↓
길이 100,000짜리 벡터
↓
대부분 0
```

문제:

```text
메모리 증가
불필요한 0 증가
고차원 벡터
```

이 때문에 대규모 범주형 데이터나 NLP에서는 Embedding을 많이 사용한다.

---

## 12. Embedding

Embedding은 단순히 shape만 바꾸는 것이 아니다.

One-Hot:

```text
고양이
→ [0,0,1,0,0,...]
```

Embedding:

```text
고양이
→ [0.24, -0.71, 0.18, 0.92, ...]
```

처럼 범주/토큰을 **학습 가능한 실수 벡터**로 표현한다.

```text
One-Hot
= 클래스 위치 표현

Embedding
= 학습된 벡터 표현
```

---

## 13. np.argmax()

Softmax 출력:

```text
[0.7, 0.2, 0.1]
```

에서 가장 큰 값의 위치를 찾는다.

```python
np.argmax([0.7, 0.2, 0.1])
# 0
```

여러 샘플:

```python
y_predict = np.argmax(
    y_predict,
    axis=1
)
```

예:

```text
[[0.7,0.2,0.1],
 [0.1,0.8,0.1],
 [0.1,0.2,0.7]]

↓

[0,1,2]
```

One-Hot된 y_test도 Accuracy 계산 전에:

```python
y_test = np.argmax(
    y_test,
    axis=1
)
```

로 class 번호로 바꿀 수 있다.

---

## 14. 회귀 / 이진분류 / 다중분류 비교

| 구분 | 회귀 Regression | 이진분류 Binary | 다중분류 Multi-class |
|---|---|---|---|
| 목표 | 연속적인 숫자 | 2개 class | 3개 이상 class |
| y 예시 | `153.7` | `0`, `1` | `0`, `1`, `2` |
| 출력층 | `Dense(1)` | `Dense(1)` | `Dense(class 수)` |
| Activation | Linear | Sigmoid | Softmax |
| 출력 예시 | `154.7` | `0.83` | `[0.1,0.7,0.2]` |
| Loss | MSE, MAE | Binary Crossentropy | Categorical Crossentropy |
| 평가 | RMSE, MAE, R² | Accuracy 등 | Accuracy 등 |
| 최종 변환 | 없음 | Threshold | Argmax |
| One-Hot | X | 보통 X | 선택 가능 |
| stratify | 보통 X | 자주 사용 | 자주 사용 |

핵심:

```text
회귀
Dense(1)
↓
Linear
↓
MSE
↓
RMSE / R²
```

```text
이진분류
Dense(1)
↓
Sigmoid
↓
Binary Crossentropy
↓
Threshold
↓
0 / 1
```

```text
다중분류
Dense(class 수)
↓
Softmax
↓
Categorical Crossentropy
↓
Argmax
↓
class 번호
```

---


> **다중분류에서는 클래스 개수만큼 출력 노드를 만들고 Softmax로 클래스별 확률 분포를 만든다. 정답을 One-Hot Encoding했다면 categorical_crossentropy를 사용하고, 최종 예측 클래스는 argmax로 결정한다.**

