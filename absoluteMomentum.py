""" 
<3.듀얼 모멘텀 전략>
= 듀얼 모멘텀 전략은 투자 자산 가운데 상대적으로 상승 추세가 강한 종목에 투자하는 상대 모멘텀 전략과
  과거 시점 대비 현재 시점의 절대적 상승세를 평가한 절대 모멘텀 전략을 결합해 위험을 관리하는 전략
- '모멘텀' 
  > 물질의 운동량이나 가속도를 의미하는 물리학 용어
  > 투자 분야에서는 주가가 한 방향성을 유지하려는 힘으로 통용
- 2009년 메반 파베르가 하버드/예일 사학 재단 기금 운용팀 높은 수익률의 비밀
- 게리 안토나치의 '듀얼 모멘텀 투자 전략'에서 공식화
- 2008년 금융위기 이후 시장의 리스크를 최소화 하면서 장기적으로 안정적인 수익을 얻을 수 있는
  다양한 자산 배분 전략과 투자 기법들이 개발되면 많은 주목
- 절대 모멘텀 전략 : 최근 N 개월 간 수익률이 양수이면 매수하고, 음수이면 공매도
- 상대 모멘텀 전략 : 투자 종목군이 10개 종목이라 할 때 10개 종목의 최근 N개월 모멘텀을
  계산해 상대적으로 모멘텀이 높은 종목을 매수하고 상대적으로 낮은 종목은 공매도

"""

### 3.1. 듀얼 모멘터 전략 구현을 위한 절대 모멘텀 전략 ###
# 현실적인 조건을 고려해서 최근 N개월 간 수익률이 양수이면 매수하고, 음수이면 매도하는
# long only 포지션으로 절대 모멘텀 전략을 구현해 보자.
# 모멘텀 지수를 계산하기 위해 과거 12개월 간 종가의 수익률을 절대 모멘텀 지수로 계산해
# 주가 변동이 0% 이상이면 매수 신호가 발생하고, 0% 미만이면 매도 신호가 발생하는 코드 작성

# 관련 패키지/툴 임포트
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from pandas import Series, DataFrame
import sqlite3
from pykrx import stock
from pykrx import bond
import exchange_calendars as ecals
import matplotlib.pyplot as plt

# 데이터 임포트
#"삼성전자" 종목에 대해 창립 이후 주가데이터 DataFrame 생성
read_df = stock.get_market_ohlcv("20100101", "20211230", "005930")
print(read_df.head(10))

##거래일이 인덱스로 되어 있어서 이 부분을 해제하고 별도의 인덱스를 생성
read_df.reset_index(drop=False, inplace=True)
print(read_df.head(10))

#describe() 함수를 사용해 데이터셋의 분포, 중심, 경향 분삭 및 형태를 파악
print(read_df.describe())

#불러온 데이터프레임에 결측치가 있다면 삭제
read_df[read_df.isin([np.nan, np.inf, -np.inf]).any(1)]

#날짜와 종가 데이터만 사용하기 위해 데이터 프레임 슬라이싱
#데이터 슬라이싱을 통해 날짜, 종가, 등락률만 선택/추출
price_df = read_df[['날짜', '종가','등락률']].copy()
print(price_df.head(10))

#이 전략에서는 월말 데이터를 사용해야 하는데 이를 위해 데이터프레임을 조작해 보자
#map() 함수를 사용하여 컬럼을 하나 추가하고 여기에 연도-월 로만 구성된 데이터를 할당한다.
price_df['STD_YM']= price_df['날짜'].map(\
    lambda x : datetime.strftime(x, '%Y-%m'))
print(price_df.head(10))

#추가된 연도-월 컬럼을 기준으로 월말 종가에 접근하는 데이터프레임을 만들어 보자
#먼저 현재 데이터프레임을 기준으로 중복을 제거한 '연-월' 데이터를 리스트에 저자한다.
month_list = price_df['STD_YM'].unique()
month_last_df = pd.DataFrame()

#별도로 작성된 데이터 프레임에 월말 조가 데이터 리스트를 루핑하면서 index[-1]을 통해
#월말 데이터에 접근하고, 데이터를 추출해 새로 만든 데이터 프레임에 적재한다.
for m in month_list:
    # 기준년월에 맞는 인덱스의 가장 마지막 날짜 row를 데이터 프레임에 추가한다
    # 이 스크립트 부분에서 Warning이 발생하지만 문제는 없다. 그냥 진행한다.
    # 향후 Pandas가 업데이트 되면 또 바꿔줘야 한다. concat로...젠장
    month_last_df = month_last_df.append(\
        price_df.loc[price_df[price_df['STD_YM'] == m].index[-1], : ])

month_last_df.set_index(['날짜'],inplace=True)
print(month_last_df.head())

#데이터 가공
# 모멘텀 지수를 계산하기 위하여 이전 시점의 데이터를 어떻게 가공했는지 먼저 확인한다.
# shift() 함수를 사용하여 BF_1M_close 컬럼에는 1개월 전 말일자 종가를 넣고
# BF_12M_close 컬럼에는 12개월 전 말일자 종가를 넣자.
month_last_df['BF_1M_close'] = month_last_df.shift(1)['종가']
month_last_df['BF_12M_close'] = month_last_df.shift(12)['종가']
month_last_df.fillna(0, inplace=True)
print(month_last_df.head(15))

#포지션 기록 : 거래장부 생성
# 이제 모멘텀 지수를 계산해서 거래가 새길 때 포진션을 기록할 데이터프레임을 만들어 둔다.
# 이 거래장부는 최종적으로는 기록된 포지션을 바탕으로 최종 수익률을 계산하는데 사용
# 처음에 만든 일별 종가가 저장된 데이터프레임을 복사해 거래 내역을 저자할 trade 컬럼 추가
book = price_df.copy()
book.set_index(['날짜'], inplace=True) #날짜 컬럼을 다시 인덱스로 지정
book['trade'] = '' #새로운 컬럼 추가
print(book.head())

#거래 실행
# 모든 전략이 그렇지만, 절대 모멘텀 전략의 핵심을 바로 '거래 실행'' 이다.

# Trading 부분
ticker = '삼성전자'
for x in month_last_df.index:
    signal = ''
    
    #절대 모멘텀을 계산
    # 월별 인덱스를 순회하면서 12개월 전 종가 대비 1개월 전 종가 수익률이 얼마인지 계산
    # 계산된 수익률은 momentum_index 변수에 저자해 0 이상인지 확인하고
    # 0 이상이면 모멘텀 현상이 나타난 것으로 판단해 매수신호가 발생하도록 한다.
    # 이 부분은 flag 변수에 저장/처리하고 추가로 다른 조거늘 넣을 수 있도록 and 조건을 걸고
    # True로 남겨 놓자. 이어서 signal 변수에 저장된 매수신호를 거래장부 데이터프레임에 저장
    # 월말 종가를 기준으로 매수/매도 신호를 계산하므로 최소 1개월 이상 해당 포지션을 유지한다.
    # 포지션을 유지하는 기간은 개인마다 다르지만 보통 1개월로 추산해 본다.
    # 이를 전문용어로 리밸런스(rebalance) 주기라고 한다.
    
    momentum_index = month_last_df.loc[x,'BF_1M_close'] / month_last_df.loc[x,'BF_12M_close'] -1
    
    # 절대 모멘텀 지표 True / False를 판단한다.
    flag = True if ((momentum_index > 0.0) and (momentum_index != np.inf) and (momentum_index != -np.inf))\
    else False \
    and True
    if flag :
        signal = 'buy ' + ticker # 절대 모멘텀 지표가 Positive이면 매수 후 보유.
    print('날짜 : ',x,' 모멘텀 인덱스 : ',momentum_index, 'flag : ',flag ,'signal : ',signal)
    book.loc[x:,'trade'] = signal
    
print(book.tail(25))

### 전략수익률 벡테스팅
# 포지션을 기록했으니 전략의 수익률을 확인해야 함
# 수익률 계산은 별도의 함수로 작성해 보자
def returns(book, ticker):
    
    #손익계산
    rtn= 1.0
    book['return'] = 1
    buy = 0.0
    sell = 0.0
    
    for i in book.index:
        if book.loc[i, 'trade'] == 'buy '+ ticker and book.shift(1).loc[i,'trade'] == '' :     # long 진입
            buy = book.loc[i, '종가']
            print('진입일 : ',i, 'long 진입가격 : ', buy)
        elif book.loc[i, 'trade'] == 'buy '+ ticker and book.shift(1).loc[i,'trade'] == 'buy '+ ticker :
            # 보유중  
            current = book.loc[i, '종가']
            rtn = (current - buy) / buy + 1
            book.loc[i, 'return'] = rtn
            
        elif book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == 'buy '+ticker:     # long 청산
            sell = book.loc[i, '종가']
            rtn = (sell - buy) / buy + 1 # 손익 계산
            book.loc[i, 'return'] = rtn
            print('청산일 : ',i, 'long 진입가격 : ', buy, ' |  long 청산가격 : ', \
                  sell, ' | return:', round(rtn, 4))

        if book.loc[i, 'trade'] == '':     # zero position
            buy = 0.0
            sell = 0.0
            current = 0.0

    acc_rtn = 1.0
    for i in book.index:
        if book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == 'buy '+ticker:     # long 청산시
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn * rtn # 누적수익률 계산
            book.loc[i:, 'acc return'] = acc_rtn

    print ('Accunulated return :', round(acc_rtn, 4))
    return (round(acc_rtn, 4))

# 함수 실행 : 실행의 결과로 1이 나오면 100% 수익, 10은 1000% 수익을 의미
returns(book, ticker)

        
       