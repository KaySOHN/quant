import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from pandas import Series, DataFrame
import sqlite3
from pykrx import stock
from pykrx import bond
import exchange_calendars as ecals
import matplotlib.pyplot as plt


#샘플로 금양(001570)의 해당 기간 내 OHLCV를 추출해 놓고
df = stock.get_market_ohlcv("20120101", "202211130", "001770")
print(df.head(10))

#불러온 데이터프레임에 결측치가 있다면 삭제
df[df.isin([np.nan, np.inf, -np.inf]).any(1)]

""" 
#그래프 그려보기
xs=df.index.to_list()					#플롯할 데이터 모두 list로 저장
ys_open=df['시가'].to_list()
ys_close=df['종가'].to_list()
ys_volume=df['거래량'].to_list()

plt.figure(figsize=(10, 8))					#전체 그래프 크기 설정

plt.subplot(2,1,1)						#2행 1열에서 1번째 그래프 지정
plt.plot(xs, ys_open, 'o-', ms=3, lw=1, label='open')		#xy데이터 플롯-line
plt.plot(xs, ys_close, 'o-', ms=3, lw=1, label='close')		#xy데이터 플롯	
#plt.ylim(300000, 1000000)					#y축 최대, 최소값
plt.yscale('log')						#y축 로그스케일 설정
plt.xlabel('Date')						#x축 이름 
plt.ylabel('Price * 10^7 (KRW)')				#y축 이름
plt.legend()							#범례 표시

plt.subplot(2,1,2)						#2행 1열에서 2번재 그래프 지정
plt.bar(xs, ys_volume, color='grey', label='volume')		#xy데이터 플롯-bar
plt.xlabel('Date')						#x축 이름
plt.ylabel('Volume')						#y축 이름
plt.legend()							#범례 표시
plt.show()

"""


#데이터 슬라이싱
price_df = df[['종가','등락률']].copy()
print(price_df.head(10))

price_df['종가'].plot(figsize=(16,9))
plt.title("종가 변화 그래프")
plt.show()

#일별 수익률 계산 = 일별 등락률을 의미하는 값이구먼 -> 이 부분은 등락률 컬럼을 이용하면 됨.
#일별 수익률을 계산하는 이유는 매수한 시점부터 매도한 시점까지 보유하게 되는데
#일별 수익률을 누적 곱하면 최종적으로 매수한 시점 대비 매도한 시점의
#종가 수익률로 계산되기 때문이다.

#일별 수익률은 원래 이렇게 해야 하는 건데
price_df['daily_rtn'] = price_df['종가'].pct_change()
#여기에 100을 곱한 PYKRX의 등락률을 그대로 복사해서 사용해 보자..그대로 사용하면 안되네 ㅎㅎ
#price_df['daily_rtn'] = price_df['등락률']
print(price_df.head(10))

#바이앤 홀드 전략의 누적곱을 계산
#판다스 cumprod() 함수를 사용하면 쉽게 계산 가능 
price_df['st_rtn'] = (1+price_df['daily_rtn']).cumprod()
print(price_df.head(10))

price_df['st_rtn'].plot(figsize=(16,9))
plt.title("바이앤홀드 전략 누적곱 그래프")
plt.show()

#특정 시점을 기준으로 수익률 계산해 보기
first_date = price_df.index[0]
last_date = price_df.index[-1]

v_pct = (price_df.loc[last_date,'종가'] / price_df.loc[first_date,'종가'])
print('수익률', v_pct)
print(price_df.tail(10))

#누적수익 구해보기
last_date = price_df.index[-1]
print('누적 수익 : ', price_df.loc[last_date, 'st_rtn'])
price_df['st_rtn'].plot(figsize=(16,9))
plt.title("바이앤홀드 전략 누적수익 그래프")
plt.show()

#특정시점을 기준으로 수익률 계산해 보기
#임의로 날짜를 지정해 보세
base_date= '2015-06-30'
tmp_df = price_df.loc[base_date:,['st_rtn']] / price_df.loc[base_date ,['st_rtn']]
last_date = tmp_df.index[-1]
print('지정일자 간 누적 수익 : ',tmp_df.loc[last_date,'st_rtn'])
tmp_df.plot(figsize=(16,9)) 
plt.title("바이앤홀드 지정일자 간 누적수익 그래프")
plt.show()

#백테스팅 및 최대 낙폭 구하기
historical_max = price_df['종가'].cummax()
daily_drawdown = price_df['종가'] / historical_max - 1.0
historical_dd = daily_drawdown.cummin()
historical_dd.plot()
plt.title("바이앤홀드 백테스팅 결과")
plt.show()

print(price_df.head(10))

### 투자성과 분석지표 ###
#1.연평균 복리 수익률(CAGR)
#보유한 데이터의 마지막 날짜를 넣어준다.
CAGR = price_df.loc['2022-11-11','st_rtn'] ** (252./len(price_df.index)) -1

#2.샤프지수
#샤프지수는 위험 대비 수익성 지표
# (실현 수익률의 산술평균 / 실현 수익률의 변동성)으로 계산
Sharpe = np.mean(price_df['daily_rtn']) / np.std(price_df['daily_rtn']) * np.sqrt(252.)

#3.변동성
#변동성은 금융 자산의 방향성에 대한 불확실성과 가격등락에 대한 위험 에상지표로
#수익률의 표준편차를 변동성으로 계산
VOL = np.std(price_df['daily_rtn']) * np.sqrt(252.)

#4.최대 낙폭(MDD, Maximum DrawDown)
#최대 낙폭은 투자기간에 고점부터 저점까지 떨어진 낙폭 중 최대값
MDD = historical_dd.min()

print('=== 선택종목의 성과 분석 결과 ===')
print('연평균 복리 수익률(CAGR) : ',round(CAGR*100,2),'%')
print('샤프 지수(Sharpe) : ',round(Sharpe,2))
print('변동성(VOL) : ',round(VOL*100,2),'%')
print('최대낙폭(MDD) : ',round(-1*MDD*100,2),'%')
