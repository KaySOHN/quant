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

### 2.1. 볼린저 밴드 기법 ###
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

#"삼천리" 종목에 대해 창립 이후 주가데이터 DataFrame 생성
df = stock.get_market_ohlcv("20000101", "20211230", "005930")
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

#특정 기간을 설정하여 볼린저 밴드를 활용한 전략의 성과를 확인해 보기
base_date = '2017-01-03'
sample = bollinger.loc[base_date:]
print(sample.head(10))

sample.plot(figsize=(16,9))
plt.title("볼린저 밴드 샘플 그래프")
plt.show()

#거래날짜를 인덱스로 설정해 놨기 때문에 loc 인덱서를 활용하면 임의로 정한
#날짜를 이용해 데이터프레임 자르기 가능

#다음은 평균 회귀 전략에서 진입/청산 신호가 발생할 때 취할 행동을 기록할
#데이터 프레임(거래장부; Trading Book)를 작성해 보자
#book 변수를 사용해 거래내역 컬럼을 추가
book = sample[['종가']].copy()
book['trade'] = ''
print(book.head(10))

#거래장부(Trade Book)을 만들 수 있는 코드를 함수화 해 놓자^^
def create_trade_book(sample):
  book = sample[['종가']].copy()
  book['trade'] = ''
  return (book)

#볼린저밴드 및 거래장부 함수를 이용해서 볼린저밴드를 활용한 전략 알고리즘 작성
### 2.5. 볼린저 밴드 거래 전략 ###
#진입조건에 맞으면 매수하고 청산조건에 맞으면 매도
#평균 회귀의 기본적인 가정 : 현재 주가가 일정기간 평균가격보다 낮다면 미래에는
#주가가 상승할 것으로 예상해 주식을 매수하고, 반대로 현재 주가가 평균가격보다 높다면
#앞으로 주가가 하락할 것으로 예상해 주식을 매도한다.
#20일 이동평균선을 기준으로 위아래라로 표준편차 두 구간을 사용하였기에 통계적으로
#95%의 주가가 밴드 내에 존재한다고 가정하에 하한선을 크게 밑도는 겨우 고려하는 전략

def tradings(sample, book):
  for i in sample.index:
    if sample.loc[i, '종가'] > sample.loc[i, 'ub']: #상단 밴드 이탈 시 동작 안함
      book.loc[i, 'trade'] = ''
    elif sample.loc[i, 'lb'] > sample.loc[i, '종가']: #하단 밴드 이탈 시 "매수"
      if book.shift(1).loc[i, 'trade'] == '매수': #이미 매수 상태라면
        book.loc[i, 'trade'] = '매수' #매수 상태 유지
      else:
        book.loc[i, 'trade'] = '매수'
    elif sample.loc[i, 'ub'] >= sample.loc[i, '종가'] and sample.loc[i, '종가'] >= sample.loc[i, 'lb']:
      #종가가 볼린저밴드 안에 있을 때는
      if book.shift(1).loc[i, 'trade'] == '매수': #매수 상태라면
        book.loc[i, 'trade'] = '매수' #매수상태 유지
      else:
        book.loc[i, 'trade'] = '' #동작 안함
        
  return (book)

book = tradings(sample, book)
print(book.tail(10))


#볼린저밴드 거래 내역을 기록했으니 수익률 계산 백테스팅
### 2.6. 볼린저 밴드 거래 전략 수익률 ###
#return() 함수를 구현해 트레이딩 book에 적혀 있는 거래 내역대로 진입/청산 일자에 따른
#매수/매도 금액응ㄹ 바탕으로 수익률을 계산, 그 다음에 계산된 수익률로 누적수익룰 계산

def returns(book):
    # 손익 계산
    rtn = 1.0
    book['return'] = 1
    buy = 0.0
    sell = 0.0
    
    for i in book.index:
        #진입조건일 경우 처리
        if book.loc[i, 'trade'] == '매수' and book.shift(1).loc[i, 'trade'] == '':     # long 진입
            buy = book.loc[i, '종가']
            print('진입일 : ',i, 'long 진입가격 : ', buy)
            
        #청산조건일 경우 처리    
        elif book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == '매수':     # long 청산
            sell = book.loc[i, '종가']
            rtn = (sell - buy) / buy + 1 # 손익 계산
            book.loc[i, 'return'] = rtn
            print('청산일 : ',i, ' 진입가격 : ', buy, ' | 청산가격 : ', \
                  sell, ' | return:', round(rtn, 4))
    
        if book.loc[i, 'trade'] == '':     # zero position
            buy = 0.0
            sell = 0.0
    
    #누적수익률 구하기
    acc_rtn = 1.0
    for i in book.index:
        rtn = book.loc[i, 'return']
        acc_rtn = acc_rtn * rtn  # 누적 수익률 계산
        book.loc[i, 'acc return'] = acc_rtn

    print ('Accunulated return :', round(acc_rtn, 4))
    return (round(acc_rtn, 4))

print(returns(book))

#해당 결과 백테스팅 그래프로 그려보자.
book['acc return'].plot(figsize=(16,9))
plt.title("볼린저 밴드 누적수익률 변화 추이 그래프")
plt.show()

print(book.tail(10))

### 결론 ###
#평균 회귀 전략이 모든 종목에 적중하는 것은 아니다. 모든 종목에 일괄적으로 적용되어
#좋은 성과를 내는 알고리즘은 희귀하다. 하락하던 주가가 별안간에 파산해 버린다면
#다시 평균으로 회귀할 수 없고, 떨어져 있던 주가가 어느 정도 상승해 팔았는데
#뜻밖의 장기적 추세가 형성돼 장기적 고공행진을 한다면 난리 ㅎㅎ
#따라서 자기만의 투자 철학과 면밀하게 파악한 시장 동향을 기반으로 전략을 수립하고
#실행해야 한다.











