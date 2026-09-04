# Day 04 - RMSE, R2, Pandas, Data Preprocessing

## 학습 내용

- MSE / RMSE 차이
- R2 회귀 평가 지표
- 회귀와 분류 차이
- Backpropagation / Gradient 복습
- Pandas DataFrame
- CSV 데이터 불러오기
- 결측치 확인 및 제거
- feature와 target 분리
- train_test_split
- Dacon 따릉이 데이터 회귀 모델 실습

## Regression Evaluation

MSE
- 예측값과 실제값의 제곱 오차 평균
- 학습 loss로 사용

RMSE
- MSE에 square root 적용
- 실제 target과 동일한 단위로 해석 가능

R2
- 회귀 모델이 데이터를 얼마나 잘 설명하는지 확인하는 보조 지표

## Pandas

```python
train_csv = pd.read_csv("train.csv")

train_csv.info()
train_csv.isnull().sum()
train_csv.dropna()