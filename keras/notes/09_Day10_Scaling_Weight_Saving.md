# Day 10 - Scaling과 Weight Saving

## 1. 오늘 학습 핵심

Day 10에서는 모델 학습 전에 feature들의 숫자 범위를 맞추는 **Scaling**과,
학습이 끝난 모델의 **가중치(Weight)를 저장하고 다시 불러오는 방법**을 학습했다.

오늘 배운 주요 내용:

```text
Scaling이 필요한 이유
↓
StandardScaler
↓
MaxAbsScaler
↓
RobustScaler
↓
Scaler별 차이
↓
fit / transform 구분
↓
Weight Saving
↓
Weight Loading
```

---

## 2. 왜 Scaling을 하는가?

머신러닝/딥러닝 데이터에서는 feature마다 숫자 크기가 크게 다를 수 있다.

예:

```text
temperature = 25
humidity    = 70
visibility  = 2000
pm10        = 35
```

이 상태에서는 어떤 feature는 수십 단위이고,
어떤 feature는 수천 단위이다.

이렇게 feature의 숫자 크기가 크게 다르면 모델 학습 과정에서
Gradient 계산과 Weight Update가 불안정하거나 비효율적일 수 있다.

따라서 Scaling을 통해 feature들의 크기를 비슷한 수준으로 맞춰준다.

중요:

> 모든 Scaler가 데이터를 0~1로 만드는 것은 아니다.

```text
MinMaxScaler     → 보통 0~1
StandardScaler   → 평균 0, 표준편차 1
MaxAbsScaler     → 보통 -1~1
RobustScaler     → 중앙값과 IQR 기준
```

---

## 3. fit과 transform

Scaler에서 중요한 개념은 `fit`과 `transform`이다.

```python
scaler.fit(x_train)
```

`fit()`은 Train 데이터를 보고 Scaling에 필요한 기준값을 계산한다.

예를 들어 StandardScaler라면:

```text
각 feature의 평균
각 feature의 표준편차
```

를 계산한다.

그 다음:

```python
x_train = scaler.transform(x_train)
```

은 계산된 기준으로 실제 데이터를 변환한다.

`fit()` + `transform()`을 한 번에 실행할 수도 있다.

```python
x_train = scaler.fit_transform(x_train)
```

---

## 4. Train에만 fit 하는 이유

정석:

```python
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```

또는:

```python
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

Test 데이터에는 다시 `fit()`하지 않는다.

잘못된 예:

```python
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)   # X
```

Test 데이터는 최종 평가용 데이터이므로,
Test 데이터의 통계 정보를 학습 과정에 이용하면 안 된다.

따라서:

```text
Train
→ 기준값 계산(fit)
→ 변환(transform)

Test
→ Train에서 만든 기준으로만 변환(transform)
```

한다.

---

## 5. StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

StandardScaler는 각 feature를:

```text
평균 ≈ 0
표준편차 ≈ 1
```

이 되도록 변환한다.

공식:

```text
z = (x - 평균) / 표준편차
```

예를 들어:

```text
원본
10, 20, 30
```

평균이 20이고 표준편차가 약 8.16이라면:

```text
10 → 약 -1.225
20 → 0
30 → 약 1.225
```

즉 StandardScaler는 0~1로 만드는 것이 아니라
**데이터의 중심을 0으로, 퍼짐을 표준편차 1로 맞추는 방식**이다.

---

## 6. MaxAbsScaler

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

MaxAbsScaler는 각 feature를
**그 feature의 절댓값 최대값으로 나눈다.**

공식:

```text
x' = x / max(|x|)
```

예:

```text
원본
-10, 20, 30
```

절댓값 최대값:

```text
30
```

변환:

```text
-10 → -0.333
20  →  0.667
30  →  1
```

따라서 결과는 일반적으로:

```text
-1 ~ 1
```

범위에 들어온다.

### 특징

```text
0은 그대로 0으로 유지
음수도 유지 가능
Sparse Data처럼 0이 많은 데이터에 유용할 수 있음
```

---

## 7. RobustScaler

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

RobustScaler는 **이상치(Outlier)에 상대적으로 덜 민감한 Scaling 방법**이다.

StandardScaler는 평균과 표준편차를 사용하지만,
RobustScaler는:

```text
Median = 중앙값
IQR = Q3 - Q1
```

을 사용한다.

공식:

```text
x' = (x - median) / (Q3 - Q1)
```

여기서:

```text
Q1 = 25% 지점
Q3 = 75% 지점
IQR = Q3 - Q1
```

예를 들어:

```text
10, 11, 12, 13, 1000
```

처럼 이상치 `1000`이 있을 경우
평균과 표준편차는 크게 영향을 받을 수 있다.

하지만 중앙값과 IQR은 상대적으로 이상치의 영향을 덜 받는다.

---

## 8. Scaler 비교

| Scaler | 기준 | 결과 특징 | 이상치 영향 |
| --- | --- | --- | --- |
| StandardScaler | 평균, 표준편차 | 평균 0 / 표준편차 1 | 큼 |
| MinMaxScaler | 최소값, 최대값 | 보통 0~1 | 매우 큼 |
| MaxAbsScaler | 절댓값 최대값 | 보통 -1~1 | 큼 |
| RobustScaler | 중앙값, IQR | 중앙값 중심 | 상대적으로 작음 |

---

## 9. Scaler를 여러 개 동시에 쓰는가?

일반적으로:

```text
StandardScaler + MinMaxScaler
```

처럼 여러 Scaler를 연속해서 적용하기보다는
데이터 특성에 맞게 **하나를 선택해서 사용**한다.

예:

```text
일반적인 데이터
→ StandardScaler

0~1 범위가 필요
→ MinMaxScaler

0이 많은 Sparse Data
→ MaxAbsScaler

이상치가 많은 데이터
→ RobustScaler
```

그리고 실제 모델에서는 여러 Scaler를 각각 적용해보고
Validation 성능을 비교해 선택할 수도 있다.

---

## 10. Weight란?

뉴런의 기본 계산:

```text
y = wx + b
```

여기서:

```text
w = Weight
b = Bias
```

모델은 학습하면서 Weight와 Bias를 계속 수정한다.

```text
초기 Weight
↓
Forward
↓
Loss 계산
↓
Backpropagation
↓
Optimizer
↓
Weight Update
↓
반복
```

따라서 학습이 끝났을 때의 Weight에는
모델이 데이터에서 학습한 정보가 들어 있다.

---

## 11. Weight Saving

학습한 Weight를 파일로 저장할 수 있다.

```python
model.save_weights("my_model.weights.h5")
```

의미:

```text
모델 구조 전체를 저장하는 것이 아니라
학습된 Weight를 저장
```

쉽게 비유하면:

```text
모델 구조 = 뇌의 구조
Weight     = 학습해서 얻은 기억
```

따라서 `save_weights()`는 모델의 학습된 기억을 저장하는 것과 비슷하다.

---

## 12. Weight Loading

저장한 Weight는 다시 불러올 수 있다.

```python
model.load_weights("my_model.weights.h5")
```

단, Weight를 불러오려면 기본적으로
**저장 당시와 호환되는 모델 구조가 준비되어 있어야 한다.**

예:

```python
model = Sequential()

model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.load_weights("my_model.weights.h5")
```

흐름:

```text
같은 모델 구조 생성
↓
저장한 Weight 불러오기
↓
기존에 학습한 상태 재사용
```

---

## 13. 전체 모델 저장과 Weight 저장 차이

### Weight만 저장

```python
model.save_weights("my_model.weights.h5")
```

저장:

```text
Weight
```

나중에 같은 모델 구조를 다시 정의한 뒤:

```python
model.load_weights("my_model.weights.h5")
```

해야 한다.

### 모델 전체 저장

```python
model.save("my_model.keras")
```

전체 모델 저장은 일반적으로:

```text
모델 구조
+
학습된 Weight
+
일부 학습 설정
```

을 함께 저장한다.

---

## 14. EarlyStopping과 Weight

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

의:

```python
restore_best_weights=True
```

도 Weight 개념과 연결된다.

예:

```text
Epoch 30
val_loss = 0.20  ← 가장 좋음

Epoch 31
val_loss = 0.22

...

Epoch 40
val_loss = 0.30
→ EarlyStopping
```

이때:

```python
restore_best_weights=True
```

이면 Epoch 40의 Weight를 사용하는 것이 아니라:

```text
Epoch 30의 Weight
```

로 되돌아간다.

즉:

> 가장 좋은 Validation 성능을 냈던 시점의 Weight를 복원한다.

---

## 15. Scaling + Weight Saving 전체 흐름

```python
from sklearn.preprocessing import StandardScaler

# 1. Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. Model Training
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32
)


# 3. Weight Saving
model.save_weights(
    "my_model.weights.h5"
)
```

나중에:

```python
model.load_weights(
    "my_model.weights.h5"
)

result = model.evaluate(
    x_test,
    y_test
)

y_predict = model.predict(
    x_test
)
```

