# Furiosa AI Study - Day 1~2

## 1. 개발 환경 구성

### Miniconda / Python 3.11

```bash
conda env list
conda create -n py311 python=3.11
conda activate py311
conda deactivate
```

Windows CMD에서 필요할 경우:

```bash
conda.bat activate py311
```

설치된 패키지 확인:

```bash
pip list
```

TensorFlow 설치:

```bash
pip install tensorflow
```

### VS Code 설정

- VS Code 설치
- `Python: Select Interpreter`에서 `py311` 환경 선택
- `Ctrl + F5` : 디버깅 없이 Python 파일 실행
- Python은 들여쓰기가 문법에 포함되므로 줄 간격에 주의
- 여러 줄을 선택하고 `Ctrl + /` : 주석 처리 / 해제
- 한 줄에 커서를 두고 `Ctrl + C`, `Ctrl + V`로 빠르게 줄 복사 가능

---

# Day 1

## 2. 인공지능 기본 구조

큰 범주로 보면 다음과 같이 이해할 수 있다.

```text
Artificial Intelligence
        ↓
Machine Learning
        ↓
Deep Learning
        ↓
Transformer / LLM 등
```

TensorFlow와 PyTorch는 딥러닝 모델을 만들고 학습시키기 위한 대표적인 프레임워크이다.

현재 수업에서는 TensorFlow 2.x와 Keras부터 시작한다.

---

## 3. 머신러닝의 가장 기본적인 목표

가장 단순한 선형식:

\[
y = ax + b
\]

딥러닝에서는 보통 다음과 같이 표현한다.

\[
y = wx + b
\]

- `w` : weight, 가중치
- `b` : bias, 편향
- 목표 : **정답과 예측값의 오차를 최소화하는 최적의 weight와 bias를 찾는 것**

즉, 학습은 결국 **loss가 작아지도록 weight를 반복해서 수정하는 과정**이다.

---

## 4. Keras 기본 구조

### keras01.py

```python
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일 / 학습
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150)

# 4. 예측
result = model.predict(np.array([4]))
print("4의 예측값 :", result)
```

### 코드 흐름

```text
데이터 준비
    ↓
모델 구성
    ↓
compile
    ↓
fit (학습)
    ↓
evaluate / predict
```

### Sequential

```python
model = Sequential()
```

레이어를 위에서 아래 순서대로 쌓는 방식이다.

### Dense

```python
Dense(...)
```

모든 입력 노드와 출력 노드가 서로 연결되는 **Fully Connected Layer(밀집층)** 이다.

---

## 5. 단층 모델과 다층 모델

단순한 모델:

```python
model = Sequential()
model.add(Dense(1, input_dim=1))
```

다층 모델:

```python
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))
```

입력 차원은 첫 번째 레이어에만 지정하면 이후 레이어에서는 일반적으로 생략할 수 있다.

예를 들어:

```text
입력 1
 ↓
Dense 3
 ↓
Dense 5
 ↓
Dense 4
 ↓
출력 1
```

---

## 6. Input / Hidden / Output Layer

```text
Input Layer
    ↓
Hidden Layer
    ↓
Hidden Layer
    ↓
Output Layer
```

- Input Layer : 입력 데이터의 feature를 받음
- Hidden Layer : 데이터의 패턴을 학습
- Output Layer : 최종 결과 출력

예:

```text
1 - 3 - 5 - 4 - 1
```

의미:

- 입력 feature : 1개
- 첫 번째 hidden layer : 3개 node
- 두 번째 hidden layer : 5개 node
- 세 번째 hidden layer : 4개 node
- 출력 : 1개

Hidden layer의 개수와 node 수는 사람이 직접 정하는 **하이퍼파라미터**이다.

---

## 7. Loss를 줄이기 위한 기본 조절 요소

초기 수업에서 조절해볼 수 있는 요소:

- Layer 개수
- Node 개수
- Epoch 수
- Batch size
- Optimizer
- Learning rate 등

이러한 값을 조절해 좋은 성능을 찾는 과정을 **Hyperparameter Tuning**이라고 한다.

단, 단순히 레이어와 epoch를 무조건 늘린다고 좋은 모델이 되는 것은 아니다.

너무 학습 데이터에만 잘 맞으면 **Overfitting(과적합)** 이 발생할 수 있다.

---

## 8. Epoch와 Batch

### Epoch

전체 학습 데이터를 한 번 모두 사용해서 학습하는 것을 1 epoch라고 한다.

```python
model.fit(x, y, epochs=100)
```

→ 전체 데이터를 100번 반복해서 학습

### Batch

전체 데이터를 여러 묶음으로 나누어 학습하는 단위이다.

```python
model.fit(
    x,
    y,
    epochs=100,
    batch_size=32
)
```

예를 들어 데이터가 100,000개라면 이를 한 번에 모두 계산하지 않고 작은 batch로 나눠서 학습할 수 있다.

Batch를 사용하는 이유:

- 메모리 사용량 감소
- 학습 효율 향상
- Gradient 업데이트 횟수 증가
- 데이터가 큰 경우 학습 가능

Keras의 `batch_size` 기본값은 일반적으로 32이다.

---

# Day 2

## 9. Tensor란?

TensorFlow의 Tensor는 다차원 배열 형태의 데이터를 의미한다.

| 차원 | 이름 | 예시 | Shape |
|---|---|---|---|
| 0차원 | Scalar | `3` | `()` |
| 1차원 | Vector | `[1,2,3]` | `(3,)` |
| 2차원 | Matrix | `[[1,2],[3,4]]` | `(2,2)` |
| 3차원 이상 | Tensor | 이미지, 영상 등 | `(2,3,4)` 등 |

딥러닝 모델에서는 데이터가 대부분 Tensor 형태로 처리된다.

---

## 10. Shape 이해하기

예:

```python
x = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print(x.shape)
```

결과:

```text
(2, 5)
```

의미:

- 2행
- 5열

머신러닝에서 2차원 tabular 데이터는 보통 다음 구조를 사용한다.

```text
(샘플 수, feature 수)
```

즉:

```text
행 = sample
열 = feature
```

---

## 11. x와 y의 샘플 수는 같아야 한다

다음 데이터는 문제가 있다.

```python
x = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

y = np.array([1, 2, 3, 4, 5])
```

Shape:

```text
x : (2, 5)
y : (5,)
```

현재 x는 **샘플이 2개**, y는 **정답이 5개**이므로 서로 맞지 않는다.

일반적인 지도학습에서는:

```text
x의 첫 번째 차원 = y의 첫 번째 차원
```

이어야 한다.

즉 x를 `(5, 2)`로 만들거나 y를 `(2,)`로 만들어야 한다.

---

## 12. Transpose

행과 열을 바꿀 때 사용한다.

```python
x = x.T
```

또는:

```python
x = x.transpose()
```

예:

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(x.shape)
# (2, 3)

x = x.transpose()

print(x.shape)
# (3, 2)
```

---

## 13. 다중 Feature 데이터

### keras08.py

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.3,1.9],
    [9,8,7,6,5,4,3,2,1,0]
])

x = x.transpose()

y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일 / 학습
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=395, batch_size=2)

# 4. 평가
loss = model.evaluate(x, y)
print("loss :", loss)

# 5. 예측
results = model.predict(np.array([[10, 1.3, 0]]))
print("[10, 1.3, 0]의 예측값 :", results)
```

Transpose 전:

```text
x.shape = (3, 10)
```

Transpose 후:

```text
x.shape = (10, 3)
```

즉:

```text
샘플 10개
feature 3개
```

가 된다.

따라서:

```python
Dense(10, input_dim=3)
```

에서 `input_dim=3`이 된다.

---

## 14. Predict 입력 Shape

학습 데이터가:

```text
(100, 3)
```

이라면:

- 샘플 : 100개
- 각 샘플의 feature : 3개

예측 데이터 한 개를 넣을 때는:

```python
np.array([[10, 31, 211]])
```

Shape:

```text
(1, 3)
```

의미:

```text
예측할 샘플 1개
feature 3개
```

즉:

```text
학습 데이터 shape
(샘플 개수, 데이터 모양...)

예측 데이터 shape
(예측할 샘플 개수, 데이터 모양...)
```

모델이 학습할 때 받았던 feature 구조와 예측 데이터의 feature 구조가 맞아야 한다.

### 한 샘플 자체가 2×2 행렬인 경우

학습 데이터:

```text
(100, 2, 2)
```

예측 데이터 1개:

```python
np.array([
    [
        [1, 2],
        [3, 4]
    ]
])
```

Shape:

```text
(1, 2, 2)
```

맨 앞의 `1`은 batch/sample 개수이다.

---

## 15. range()

```python
range(10)
```

은:

```text
0 ~ 9
```

즉, 끝 숫자는 포함하지 않는다.

예:

```python
x = np.array([
    range(10),
    range(21, 31),
    range(201, 211)
])
```

Shape:

```text
(3, 10)
```

필요하면 transpose:

```python
x = x.T
```

결과:

```text
(10, 3)
```

---

# 16. 학습 데이터와 평가 데이터 분리

지금까지 단순 예제에서는 같은 데이터로 학습과 평가를 하는 경우가 많았다.

```python
model.fit(x, y)
model.evaluate(x, y)
```

하지만 실제 머신러닝에서는 이것만으로 모델 성능을 판단할 수 없다.

모델이 학습 데이터 자체를 외워버리는 **Overfitting** 문제가 생길 수 있기 때문이다.

따라서 데이터를 보통 다음과 같이 분리한다.

```text
전체 데이터
   ↓
Train
Validation
Test
```

예:

```text
Train      70%
Validation 15%
Test       15%
```

또는 상황에 따라:

```text
Train 70%
Test 30%
```

처럼 시작할 수도 있다.

### Train

모델의 weight를 실제로 학습하는 데이터

### Validation

학습 중 모델의 일반화 성능을 확인하고 하이퍼파라미터를 조절하는 데이터

### Test

학습이 모두 끝난 후 최종 성능을 확인하는 데이터

핵심:

> 과거 데이터를 단순히 다시 맞추는 것이 목표가 아니라, **보지 못한 새로운 데이터에서도 잘 예측하는 모델**을 만드는 것이 목표다.

---

# 17. Training과 Inference

## Training

학습은 데이터와 정답을 이용해 weight를 찾는 과정이다.

```text
입력 데이터
    ↓
모델
    ↓
예측
    ↓
정답과 비교
    ↓
Loss
    ↓
Backpropagation
    ↓
Weight 수정
```

예를 들어 이미지 AI:

```text
100만 장 이미지
+
사람 / 자동차 / 차선 정답
        ↓
      GPU
        ↓
Training
        ↓
학습된 Weight
```

학습 결과는 다음과 같은 모델 파일로 저장될 수 있다.

```text
model.pth
model.onnx
SavedModel
```

---

## Inference

추론은 **이미 학습이 끝난 모델의 weight를 이용해 새로운 입력의 결과를 계산하는 과정**이다.

```text
새로운 입력
   ↓
학습 완료 모델
   ↓
Forward 연산
   ↓
예측 결과
```

자율주행 예:

```text
Camera
  ↓
도로 영상
  ↓
학습된 AI 모델
  ↓
Inference
  ↓
자동차 / 보행자 / 차선 검출
```

### 핵심 차이

| 구분 | Training | Inference |
|---|---|---|
| 목적 | Weight 학습 | 새로운 데이터 예측 |
| 정답 데이터 | 필요 | 보통 필요 없음 |
| Loss 계산 | O | 보통 X |
| Backpropagation | O | X |
| Weight 변경 | O | X |
| 주요 연산 | Forward + Backward | Forward |

---

# 18. CPU / GPU / TPU / NPU

## CPU

범용 연산 장치.

특징:

- 복잡한 순차 연산에 강함
- GPU에 비해 코어 수가 적음
- 대규모 딥러닝 병렬 연산에는 비효율적

AI 활용:

- 데이터 전처리
- 프로그램 제어
- 작은 모델의 추론

---

## GPU

대규모 병렬 연산에 강한 장치.

특징:

- 많은 연산 코어
- 행렬 / Tensor 연산에 강함
- 딥러닝 학습과 추론 모두 널리 사용

AI 활용:

- CNN 학습
- Transformer 학습
- Object Detection
- LLM 학습 / 추론

---

## TPU

Google이 개발한 AI 가속기.

특징:

- Tensor 연산에 특화
- TensorFlow / JAX와 강한 생태계
- 대규모 학습뿐 아니라 추론에도 활용 가능

---

## NPU

Neural Processing Unit.

신경망 연산을 효율적으로 처리하기 위한 AI 전용 가속기이다.

특징:

- Tensor / Matrix 연산 특화
- 높은 처리량
- 낮은 전력 대비 높은 AI 성능을 목표로 함
- 제품에 따라 학습 지원 여부가 다르지만 **추론 가속 용도로 매우 많이 사용**

활용:

```text
학습된 모델
    ↓
모델 변환 / 최적화
    ↓
NPU
    ↓
Inference
```

퓨리오사AI 교육에서는 실제 NPU에서 모델을 실행하며 **AI 추론과 모델 최적화 경험**을 쌓는 것이 중요한 부분이다.

> 참고: NVIDIA Jetson은 일반적으로 GPU 기반 엣지 AI 플랫폼으로 분류되므로, NPU의 대표 예시와는 구분해서 이해하는 것이 좋다.

---

# 19. AI 하드웨어의 역할 정리

```text
CPU
→ 범용 처리 / 데이터 전처리 / 제어

GPU
→ 대규모 병렬 연산
→ AI 학습 + 추론

TPU
→ Google의 AI 전용 가속기
→ 대규모 Tensor 연산

NPU
→ Neural Network 특화 가속기
→ 특히 효율적인 AI 추론에 강점
```

정확한 속도는 모델, batch size, 정밀도, 하드웨어 종류에 따라 크게 달라지므로 단순히 FPS 숫자만으로 비교하면 안 된다.

---

# 20. 현재까지 중요하게 배운 내용

## 데이터

- 모델을 만들기 전에 데이터의 shape을 먼저 확인한다.
- 2차원 데이터에서는 일반적으로:
  - 행 = sample
  - 열 = feature
- x와 y의 샘플 개수는 같아야 한다.
- 필요하면 `transpose()`로 행과 열을 바꿀 수 있다.

## 모델

- `Sequential`로 레이어를 순서대로 구성
- `Dense`는 Fully Connected Layer
- 첫 레이어의 `input_dim`은 feature 개수와 맞춰야 한다.
- Layer, Node, Epoch, Batch Size 등은 하이퍼파라미터이다.

## 학습

- 학습의 목적은 최소 loss를 만드는 적절한 weight를 찾는 것
- 같은 학습 데이터에서 성능이 좋다고 실제 성능까지 좋은 것은 아니다.
- Train / Validation / Test 데이터를 분리해야 한다.

## 추론

- 학습된 모델을 이용해서 새로운 데이터를 예측
- 예측 데이터의 feature shape은 학습 데이터와 맞아야 한다.
- NPU는 이런 inference 연산을 빠르고 효율적으로 수행하기 위한 AI 가속기이다.

---

# 21. 앞으로 공부할 때 체크할 것

1. `x.shape`, `y.shape`를 항상 출력해서 확인
2. sample과 feature를 구분
3. `Dense`의 input shape이 왜 그렇게 되는지 설명할 수 있게 하기
4. `fit`, `evaluate`, `predict` 차이 이해
5. Training과 Inference 차이 이해
6. Train / Validation / Test 데이터 분리
7. Overfitting 개념
8. Tensor / Matrix 연산
9. GPU와 NPU의 역할 차이
10. 코드만 실행하지 말고 각 줄이 왜 필요한지 설명할 수 있게 공부

---

## 한 줄 요약

> **좋은 AI 모델을 만드는 핵심은 데이터를 올바른 shape으로 구성하고, 학습 데이터에서 loss만 줄이는 것이 아니라 새로운 데이터에서도 잘 동작하도록 학습시키는 것이다. 이후 학습된 모델은 GPU/NPU 같은 가속기에서 추론에 사용될 수 있다.**
