---
title: 데이터 전처리 기술 다루기 (1)
date: 2023-10-02 18:32:00 +09:00
categories: [Data Engineer, Data Preprocessing]
tags: [Data Enginner, Python, Pandas, NumPy, Data Preprocessing]
math: true
---

> **데이터 정제 (Data Cleaning)**
> 
> 데이터 정제(Data Cleaning)란, 데이터의 오류, 불일치, 불완전성 또는 무의미한 부분을 식별하고
> 이러한 문제를 해결하여 데이터의 품질을 향상시키는 과정이다.

## **결측치 처리 (Handling Missing Values)**
결측치는 누락된 데이터를 나타내며, 이를 대체, 삭제, 예측을 통해 처리하여 데이터 완성도를 유지하고 분석 가능한 형태로 만든다.

#### 1. 결측치 대체

```python
import pandas as pd
import numpy as np

data = {'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# 결측치 대체 (평균값으로 대체)
mean_a = df['A'].mean()
mean_b = df['B'].mean()
df['A'].fillna(mean_a, inplace=True)
df['B'].fillna(mean_b, inplace=True)
```

#### 2. 결측치 삭제

```python
import pandas as pd
import numpy as np

data = {'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# 결측치 삭제
df.dropna(inplace=True)
```

#### 3. 결측치 예측

결측치가 있는 행을 삭제하면 데이터의 양이 줄어들고 중요한 정보를 손실할 수 있다. 결측치를 예측하여 더 정확한 분석과 모델링을 할 수 있다.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = {
    '날짜': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    '주식 가격': [100, 105, np.nan, 110, np.nan, 120, 125, 130, np.nan, 140]
}

df = pd.DataFrame(data)

df.set_index('날짜', inplace=True)

# 결측치를 예측하기 위한 회귀 모델 학습
model = LinearRegression()

# 결측치가 아닌 데이터를 학습에 사용
X_train = df.dropna()[['주식 가격']]
y_train = df.dropna()['주식 가격']

model.fit(X_train, y_train)

# 결측치를 예측하고 채우기
X_predict = df[df['주식 가격'].isnull()][['주식 가격']]
predicted_values = model.predict(X_predict)

df.loc[df['주식 가격'].isnull(), '주식 가격'] = predicted_values

print(df)
```

```python
import pandas as pd
import numpy as np

data = {
    '날짜': pd.date_range(start='2023-10-01', periods=10, freq='D'),
    '주식 가격': [100, 105, np.nan, 110, np.nan, 120, 125, 130, np.nan, 140]
}

df = pd.DataFrame(data)

df.set_index('날짜', inplace=True)

# 선형 보간을 사용하여 결측치 채우기
df['주식 가격'] = df['주식 가격'].interpolate(method='linear')

print(df)
```

## **이상치 처리 (Handling Outliers)**
이상치는 데이터에서 벗어난 값으로, `통계적 방법을 사용하여 이상치를 감지`하고 삭제하거나 대체하여 모델의 안정성을 향상시킨다.

#### 1. 표준편차 (Standard Deviation)

표준편차(Standard Deviation)는 데이터의 산포도(분산)를 나타내는 통계적 측정 지표 중 하나다. 데이터 집합 내의 각 데이터 포인트가 평균으로부터 얼마나 퍼져 있는지, 즉 데이터의 분포가 얼마나 퍼져 있는지를 나타낸다. 표준편차가 작으면 데이터 포인트들이 평균 주변에 모여 있고, 표준편차가 크면 데이터 포인트들이 평균에서 멀리 퍼져 있음을 의미한다.

$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$

```python
import math

data = [10, 12, 14, 16, 18, 100, 22, 24, 26, 28]
mean = sum(data) / len(data)

squared_diff = [(x - mean) ** 2 for x in data]
mean_squared_diff = sum(squared_diff) / (len(data) - 1) # 자유도를 고려하여 (n-1)로 나눔
std_deviation = math.sqrt(mean_squared_diff)

print("표준 편차:", std_deviation)
```

```python
import numpy as np

data = [10, 12, 14, 16, 18, 100, 22, 24, 26, 28]

data_array = np.array(data)
std_deviation = np.std(data_array, ddof=1)  # ddof=1은 자유도를 고려

print("표준 편차:", std_deviation)
```

> 만약 모든 데이터 포인트에 대한 정보를 사용하여 표준 편차를 계산한다면 분모로 $$N$$을 사용할 수 있다.
> 그러나 표본 데이터의 경우, 데이터 포인트 중 일부만 사용하므로 이로 인해 추정된 표준 편차가 모집단의 실제 표준 편차보다 작아진다.
> 이러한 문제를 보정하기 위해 $$N-1$$로 나누어 자유도를 고려한다.
{: .prompt-tip }

#### 2. 이상치 감지

| 데이터 값 | 평균 | 표준 편차 | Z-score | 이상치 여부 (Threshold=2) |
|-----------|------|------------|---------|-----------------------------|
| 10        | 27.0 | 26.352313  | -0.64510 | No                          |
| 12        | 27.0 | 26.352313  | -0.56920 | No                          |
| 14        | 27.0 | 26.352313  | -0.49331 | No                          |
| 16        | 27.0 | 26.352313  | -0.41742 | No                          |
| 18        | 27.0 | 26.352313  | -0.34152 | No                          |
| 100       | 27.0 | 26.352313  | 2.770155 | Yes                         |
| 22        | 27.0 | 26.352313  | -0.18973 | No                          |
| 24        | 27.0 | 26.352313  | -0.11384 | No                          |
| 26        | 27.0 | 26.352313  | -0.03794 | No                          |
| 28        | 27.0 | 26.352313  | 0.037947  | No                          |


Z-score (표준 점수 또는 Z 점수) 는 개별 데이터 포인트가 평균으로부터 얼마나 표준 편차만큼 떨어져 있는지를 측정하는 표준화된 점수다. $$Z$$는 Z-score, $$X$$는 데이터 포인트의 값, $$\mu$$는 데이터 집합의 평균 그리고 $$\sigma$$는 데이터 집합의 표준 편차 (standard deviation)다.

$$Z = \frac{X - \mu}{\sigma}$$

```python
import numpy as np

# 이상치를 감지하는 함수
def detect_outliers(data, threshold=2):
    mean = np.mean(data)
    std_deviation = np.std(data, ddof=1)

    # Z-score 계산
    z_scores = [(x - mean) / std_deviation for x in data]

    # 이상치 식별
    outlier_indices = np.where(np.abs(z_scores) > threshold)
    outliers = [data[i] for i in outlier_indices[0]]

    return outliers

data = [10, 12, 14, 16, 18, 100, 22, 24, 26, 28]

detected_outliers = detect_outliers(data)

print("원본 데이터:", data)
print("감지된 이상치 값:", detected_outliers)
```

#### 3. 이상치 삭제

```python
import numpy as np
import pandas as pd

# 이상치를 감지하고 삭제하는 함수
def detect_and_remove_outliers(data, threshold=2):
    mean = np.mean(data)
    std_deviation = np.std(data, ddof=1)
    z_scores = [(x - mean) / std_deviation for x in data]

    df = pd.DataFrame({'Value': data, 'Z-Score': z_scores})
    outlier_indices = np.where(np.abs(z_scores) > threshold)

    df.drop(index=outlier_indices[0], inplace=True) # 이상치를 삭제

    return df['Value'].tolist()


data = [10, 12, 14, 16, 18, 100, 22, 24, 26, 28]

processed_data = detect_and_remove_outliers(data)

print("원본 데이터:", data)
print("이상치 삭제 후 데이터:", processed_data)
```

#### 4. 이상치 대체

```python
import numpy as np
import pandas as pd

# 이상치를 감지하고 처리하는 함수
def detect_and_process_outliers(data, threshold=2):
    mean = np.mean(data)
    std_deviation = np.std(data, ddof=1)
    z_scores = [(x - mean) / std_deviation for x in data]

    df = pd.DataFrame({'Value': data, 'Z-Score': z_scores})
    outlier_indices = np.where(np.abs(z_scores) > threshold)

    df.loc[outlier_indices[0], 'Value'] = np.nan            # 이상치를 NaN으로 대체
    df['Value'].fillna(df['Value'].median(), inplace=True)  # 결측치를 중간값(median)으로 대체

    return df['Value'].tolist()

data = [10, 12, 14, 16, 18, 100, 22, 24, 26, 28]

processed_data = detect_and_process_outliers(data)

print("원본 데이터:", data)
print("이상치 처리 후 데이터:", processed_data)
```

## **중복 데이터 처리 (Handling Duplicate Data)**
중복 데이터는 데이터 불일치를 초래하므로 중복 레코드를 식별하고 처리하여 데이터 일관성을 유지하고 분석의 정확성을 향상시킨다.

```python
import pandas as pd

data = {'이름': ['Alice', 'Bob', 'Alice', 'David', 'Bob'],
        '나이': [25, 30, 25, 35, 30],
        '도시': ['서울', '뉴욕', '서울', '로스앤젤레스', '뉴욕']}

df = pd.DataFrame(data)

print("중복 데이터 확인:")
print(df[df.duplicated()])

# 중복 데이터 제거
df = df.drop_duplicates()

print("\n중복 데이터 제거 후:")
print(df)
```