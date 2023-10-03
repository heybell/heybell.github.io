---
title: 데이터 전처리 기술 다루기 (2)
date: 2023-10-03 15:32:00 +09:00
categories: [Data Engineer, Data Preprocessing]
tags: [Data Enginner, Python, Pandas, NumPy, Data Preprocessing]
math: true
---

> **데이터 스케일링 (Data Scaling)**
>
> 다양한 특성(feature) 또는 변수(variable)들 간의 스케일(scale)을 조정하는 작업을 말한다.
> 데이터 스케일링은 데이터의 특성들을 일정한 범위나 표준화된 형태로 조정함으로써 머신러닝 모델의 학습이나 예측 성능을 향상시키는데 도움을 준다.

## **특성 간의 스케일 차이**

서로 다른 특성들이 서로 다른 단위나 스케일을 가지고 있을 때, 일부 특성이 다른 특성에 비해 모델 학습에 미치는 영향이 크게 달라질 수 있다. 이로 인해 모델이 일부 특성에 민감하게 반응하거나 일부 특성이 무시될 수 있다.

| 예시                             | 특성 1              | 특성 2           | 스케일 차이 설명                                           |
| -------------------------------- | ------------------- | ---------------- | ------------------------------------------------------------ |
| 주식 가격 예측 | 주식 가격 (달러)   | 거래량 (주)     | 주식 가격은 큰 값의 범위를 가질 수 있으며,<br> 거래량은 상대적으로 작은 값의 범위를 가진다. |
| 날씨 데이터   | 온도 (섭씨)        | 강수량 (mm)      | 온도는 일반적으로 작은 범위에 있을 수 있으며,<br> 강수량은 큰 범위에 있을 수 있다. |
| 경제 지표 데이터                  | 실업률 (%)          | GDP 성장률 (%)   | 실업률은 작은 범위에 있을 수 있으며,<br> GDP 성장률은 큰 범위에 있을 수 있다. |

* 다음은 아래 데이터를 이용하여 예제를 작성할 것이다.

```python
import numpy as np

msft_close_prices = np.array([
    328.660004, 333.549988, 332.880005, 329.910004, 334.269989,
    337.940002, 331.769989, 336.059998, 338.700012, 330.220001,
    329.059998, 328.649994, 320.769989, 319.529999, 317.010010,
    317.540009, 312.140015, 312.790009, 313.640015, 315.750000
])

tsla_close_prices = np.array([
    245.009995, 256.489990, 251.919998, 251.490005, 248.500000,
    273.579987, 267.480011, 271.299988, 276.040009, 274.390015,
    265.279999, 266.500000, 262.589996, 255.699997, 244.880005,
    246.990005, 244.119995, 240.500000, 246.380005, 250.220001
])

amzn_close_prices = np.array([
    138.119995, 137.270004, 135.360001, 137.850006, 138.229996,
    143.100006, 141.229996, 144.850006, 144.720001, 140.389999,
    139.979996, 137.630005, 135.289993, 129.330002, 129.119995,
    131.270004, 125.980003, 125.980003, 125.980003, 127.120003
])

stock_data = np.array([
    msft_close_prices,
    tsla_close_prices,
    amzn_close_prices
])
```

## **표준화(Standardization)**

평균이 0이고 표준 편차가 1인 스케일로 조정한다. 이것은 Z-점수 정규화로도 알려져 있다. 데이터 특징이 가우시안 분포를 따를 때 유용하다.

$$Z(x) = \frac{x - \mu}{\sigma}$$

```python
def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    standardized_data = (data - mean) / std
    return standardized_data

msft_standardized = standardize(msft_close_prices)
tsla_standardized = standardize(tsla_close_prices)
amzn_standardized = standardize(amzn_close_prices)
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

msft_standardized = scaler.fit_transform(msft_close_prices.reshape(-1, 1))
tsla_standardized = scaler.fit_transform(tsla_close_prices.reshape(-1, 1))
amzn_standardized = scaler.fit_transform(amzn_close_prices.reshape(-1, 1))
```

## **정규화(Normalization)**

0에서 1 사이의 범위로 스케일링한다. 이것은 최솟값-최댓값 스케일링으로도 알려져 있다. 데이터 특징이 균일한 분포를 가질 때 유용하다.

$$X_{\text{normalized}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

```python
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

msft_normalized = normalize(msft_close_prices)
tsla_normalized = normalize(tsla_close_prices)
amzn_normalized = normalize(amzn_close_prices)
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

msft_normalized = scaler.fit_transform(msft_close_prices.reshape(-1, 1))
tsla_normalized = scaler.fit_transform(tsla_close_prices.reshape(-1, 1))
amzn_normalized = scaler.fit_transform(amzn_close_prices.reshape(-1, 1))
```

## **로버스트 스케일링(Robust Scaling)**

중앙값(median)과 사분위 범위(IQR)에 기반하여 스케일링한다. 데이터에 이상치(outliers)가 포함되어 있을 때 유용하다.

$$R(x) = \frac{x - \text{median}(X)}{\text{IQR}(X)}$$


```python
def robust_scale(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    scaled_data = (data - median) / iqr
    return scaled_data

msft_robust_scaled = robust_scale(msft_close_prices)
tsla_robust_scaled = robust_scale(tsla_close_prices)
amzn_robust_scaled = robust_scale(amzn_close_prices)
```

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

msft_robust_scaled = scaler.fit_transform(msft_close_prices.reshape(-1, 1))
tsla_robust_scaled = scaler.fit_transform(tsla_close_prices.reshape(-1, 1))
amzn_robust_scaled = scaler.fit_transform(amzn_close_prices.reshape(-1, 1))
```

## **최대 절대값 스케일링(Max Absolute Scaling)**

최대 절댓값에 기반하여 스케일링한다. 데이터가 양수와 음수 값을 모두 포함하고 있을 때 유용하다.

$$M(x) = \frac{x}{\text{max}|X|}$$

```python
def max_absolute_scale(data):
    max_abs = np.max(np.abs(data))
    scaled_data = data / max_abs
    return scaled_data

msft_max_absolute_scaled = max_absolute_scale(msft_close_prices)
tsla_max_absolute_scaled = max_absolute_scale(tsla_close_prices)
amzn_max_absolute_scaled = max_absolute_scale(amzn_close_prices)
```

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

msft_max_absolute_scaled = scaler.fit_transform(msft_close_prices.reshape(-1, 1))
tsla_max_absolute_scaled = scaler.fit_transform(tsla_close_prices.reshape(-1, 1))
amzn_max_absolute_scaled = scaler.fit_transform(amzn_close_prices.reshape(-1, 1))
```

## **분위수 변환 스케일링(Quantile Transformer Scaling)**

균일 분포나 정규 분포를 따르도록 변환한다. 데이터가 가우시안 분포를 가지지 않을 때 유용하다.

#### 1. 균일 분포(Uniform distribution)

**형태**<br>
균일 분포는 모든 확률 변수가 동등한 확률로 나타날 때의 분포로, 직사각형 형태의 분포를 갖는다. 모든 값이 동등한 확률로 나타나기 때문에 균일하게 분포되어 있다. 

**확률 변수의 분포**<br>
균일 분포의 확률 변수는 모든 가능한 값에 대해 동일한 확률을 가진다. 따라서 모든 값이 나타날 확률이 동일하며, 이산적인(discrete) 또는 연속적인(continuous) 데이터에 사용될 수 있다.

**평균과 분산**<br>
균일 분포의 평균은 가능한 모든 값의 중심에 위치하며, 분산은 0.0833(1/12)으로 고정되어 있다. 모든 값이 동일한 확률을 가지기 때문에 분산이 작다.

**사용 사례**<br>
균일 분포는 난수 생성, 확률 모델링, 확률적 시뮬레이션 등에서 사용된다. 균일한 확률 분포를 가정하는 경우에 사용된다.

$$U(x) = F_X(x)$$

```python
def quantile_transform_uniform(data):
    sorted_data = np.sort(data)
    n = len(data)
    quantiles = np.arange(0, n) / (n - 1)
    transformed_data = np.interp(data, sorted_data, quantiles)
    return transformed_data

msft_quantile_uniform = quantile_transform_uniform(msft_close_prices)
tsla_quantile_uniform = quantile_transform_uniform(tsla_close_prices)
amzn_quantile_uniform = quantile_transform_uniform(amzn_close_prices)
```

```python
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(output_distribution='uniform')

msft_quantile_uniform = scaler.fit_transform(msft_close_prices.reshape(-1, 1))
tsla_quantile_uniform = scaler.fit_transform(tsla_close_prices.reshape(-1, 1))
amzn_quantile_uniform = scaler.fit_transform(amzn_close_prices.reshape(-1, 1))
```

#### 2. 정규 분포(Normal distribution)

**형태**<br>
정규 분포는 종 모양의 대칭적인 분포를 가진다. 그래프로 나타내면 가운데 평균값을 중심으로 좌우 대칭이며, 꼬리 부분이 멀어질수록 확률이 감소하는 형태를 갖는다.

**확률 변수의 분포**<br>
정규 분포의 확률 변수는 평균 주변에서 가장 자주 나타나며, 이상치(outlier)가 상대적으로 적다. 많은 자연 현상과 통계적 데이터가 정규 분포에 근사하는 경향이 있다.

**평균과 분산**<br>
정규 분포의 평균과 분산은 분포의 중심과 분포의 퍼짐 정도를 나타낸다. 평균은 분포의 중심 위치를 나타내며, 분산은 데이터가 얼마나 퍼져 있는지를 나타낸다.

**사용 사례**<br>
정규 분포는 자연 현상, 통계적 실험, 품질 관리 등 다양한 분야에서 사용된다. 데이터의 통계 분석 및 가설 검정에 자주 활용된다.

$$\text{transformed_data}(x) = \arcsin\left(\frac{i}{n-1}\right)$$

```python
def quantile_transform_normal(data):
    sorted_data = np.sort(data)
    n = len(data)
    quantiles = np.arange(0, n) / (n - 1)
    transformed_data = np.interp(data, sorted_data, quantiles)
    transformed_data = np.arcsin(transformed_data)  # Inverse sine transform
    return transformed_data

msft_quantile_normal = quantile_transform_normal(msft_close_prices)
tsla_quantile_normal = quantile_transform_normal(tsla_close_prices)
amzn_quantile_normal = quantile_transform_normal(amzn_close_prices)
```

```python
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(output_distribution='normal')

msft_quantile_normal = scaler.fit_transform(msft_close_prices.reshape(-1, 1))
tsla_quantile_normal = scaler.fit_transform(tsla_close_prices.reshape(-1, 1))
amzn_quantile_normal = scaler.fit_transform(amzn_close_prices.reshape(-1, 1))
```