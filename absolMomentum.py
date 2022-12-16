"""
[전통 퀀트 투자 전략]
- 퀀트 투자를 달리 표현하면 데이터 기반 전략 
- 머신러닝이나 딥러닝을 활용하는 투자 방법도 데이터 기반의 투자 전략
- 먼저 전통적인 투자 전략을 알아보고 이를 ML/DL 기바의 전략과 기존 퀀트 전략을 구분


<1. 전통 퀀트 방법론>
- 퀀트는 정량적 방법론을 기반으로 투자의사를 결정하는 것
- '정량적' 의 의미 : 모든 것을 수치화 하는 것
- '기술 지표 투자 전략' : 데이터에 따라 주가를 사용해 기술 지표를 만들고
  이를 투자에 활용하는 기술
  > 모멘텀 지표 : 올라가는 주식에 계속 오른다고 믿는 전략
  > 평균회귀 지표 : 올라간 주식은 반드시 평균으로 회귀한다는 전략
  > 주요 대상 : 이동평균선, 상대 강도 지수, 스토캐스틱 오실레이터 등
  > 핵심 : 주가 데이터를 활용해 각종 기술지표를 만들어 내는 것
  > 해당 지표들로 신호를 발생시켜 종목을 매매하고 전략의 수익률이나 승률을
    살펴보는 것
- '가치 투자 전략' : 기업 재무제표를 사용하는 기법
  > 당기순이익, 영업이익, 영업이익률, 매출액, 부채비율, PER, PBR, PSR,
    PCR, ROE, ROA 등 기업의 가치판단에 기준이 되는 재무제표 데이터를 기초
  > 기준이 되는 데이터에서 순위를 매겨 분위별로 자르는 작업이 중요
  > 조엘 그린블라트가 만든 '마법공식'을 구현하여 테스트
  > 자본수익률과 이익수익률로 만든 마법공식 활용

<2.평균 회귀 전략>
= '평균으로의 회귀'는 많은 자료를 토대로 결과를 예측할 때 평균에 가까워지려는
  경향성. 전체적으로 평균을 유지한다는 의미
  
"""

### 4.2.1. 볼린저 밴드 기법 ###
# 볼린저밴드는 현재 주가가 상대적으로 높은지 낮은지를 판단할 때 사용하는 지표
# 3개의 선 : 중심선인 이동평균선, 상단선과 하단선은 표준편차
# 상단밴드 : 중간밴드 + 2*20일 이동 표준 편차
# 중단밴드 : 20일 이동 평균선
# 하단밴드 : 중간밴드 - 2*20일 이동 평균 편차

# 관련 툴 임포트
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from pandas import Series, DataFrame
import sqlite3
from pykrx import stock
from pykrx import bond
import exchange_calendars as ecals
import matplotlib.pyplot as plt

#"금양" 종목에 대해 창립 이후 주가데이터 DataFrame 생성
df = stock.get_market_ohlcv("19950501", "20211230", "001770")
print(df.head(10))

##거래일이 인덱스로 되어 있어서 이 부분을 해제하고 별도의 인덱스를 생성
#df.reset_index(drop=False, inplace=True)
#print(df.head(10))

#describe() 함수를 사용해 데이터셋의 분포, 중심, 경향 분삭 및 형태를 파악
print(df.describe())

#불러온 데이터프레임에 결측치가 있다면 삭제
df[df.isin([np.nan, np.inf, -np.inf]).any(1)]

#모든 데이터를 사용할 필요가 없으므로 필요한 변수만 선택
#데이터 슬라이싱을 통해 날짜, 종가, 등락률만 선택/추출
price_df = df[['종가','등락률']].copy()
print(price_df.head(10))

""" 

###볼린저 밴드 만들기
#1) 20일 이동 평균선은 rolling()/mean() 함수를 이용하여 생성
price_df['center'] = price_df['종가'].rolling(window = 20).mean()
print(price_df.iloc[18:25])

#2) 상단 밴드 만들기
#ub라는 컬럼에 std() 함수를 이용해 표준편차를 계산
price_df['ub'] = price_df['center'] + 2 * price_df['종가'].rolling(window = 20).std()
print(price_df.iloc[18:25])

#3) 하단단 밴드 만들기
#lb라는 컬럼에 std() 함수를 이용해 표준편차를 계산
price_df['lb'] = price_df['center'] - 2 * price_df['종가'].rolling(window = 20).std()
print(price_df.iloc[18:25])

"""

#향후 사용을 위해 볼린저밴드 생성을 위한 함수를 작성
n = 20 #이동 평균선 기간
sigma = 2 #표준편차 값

def bollinger_band(price_df, n, sigma):
    bb = price_df.copy()
    bb['center'] = price_df['종가'].rolling(n).mean()
    bb['ub'] = bb['center'] + sigma + price_df['종가'].rolling(n).std()
    bb['lb'] = bb['center'] - sigma + price_df['종가'].rolling(n).std()
    return bb

bollinger = bollinger_band(price_df, n, sigma)
base_date = '2021-01-03'
sample = bollinger.loc[base_date:]
print(sample.head(10))


