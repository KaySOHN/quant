
### 3.2. 듀얼 모멘터 전략 구현을 위한 상대 모멘텀 전략 ###
# 절대 모멘텀 전략은 단일 종목으로 백테스팅 가능
# 그러나, 상대 모멘텀 전략은 다수 종목으로 투자 종목 대상군을 형성해야 이용 가능
# 상대 모멘텀 전략에서는 모멘텀 지수를 계산하기 위해 과거 1개월 종가의 수익률을 계산
# 지난 달 마지막 날을 기준으로 전체 투자 대상 종목에서 상대적으로 높은 수익률을 보인
# 상위 종목에 매수 신호를 발생시키는 전략

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
import yfinance as yf #Yahoo Finance를 사용하기 위해 새로 Import한 부분^^



### 1단계 : 데이터  추출 ###
#1) 먼저 데이터를 읽어서 저장할 목표 데이터프레임을 생성하자.
# Monthly Data 저장 데이터프레임
month_last_df = pd.DataFrame(columns=['날짜','CODE','1M_RET'])

#종목 데이터프레임 생성
stock_df = pd.DataFrame(columns=['날짜','CODE','종가'])

#불러 온 주가 데이터를 가공할 함수 data_processing 함수를 먼저 정의해 보자
def data_preprocessing(sample, ticker, base_date):   
    #sample['CODE'] = ticker # 종목코드 추가...이 부분은 이미 추가했으니 주석처리
    sample = sample[sample['날짜'] >= base_date][['날짜','CODE','종가']].copy() # 기준일자 이후 데이터 사용
    sample.reset_index(inplace= True, drop= True)
    
    # 기준년월 처리
    sample['STD_YM'] = sample['날짜'].map(lambda x : datetime.strftime(x,'%Y-%m')) 
    sample['1M_RET'] = 0.0 # 수익률 컬럼
    ym_keys = list(sample['STD_YM'].unique()) # 중복 제거한 기준년월 리스트
    print(sample.head(10))
    
    return sample, ym_keys

#2) 데이터를 가져 올 종목 코드를 리스트에 저장해서 For Loop를 돌리면서 처리
#일단 샘플로 금양, 삼천리, 삼성전자, 현대에너지솔루션, 서울가스 5개로 시작^^
tickers = ['001570','004690','005930','322000','017390']

#3) For Loop로 OHLCV 수집하여 DataFrame에 적재. 일단 2021년까지 데이터만
for ticker in tickers:
    base_date='2021-01-02' #향후 점검할 기준일 데이터 미리 설정
    read_df = stock.get_market_ohlcv("19800101", "20211230", ticker)
    #만들어진 DataFrame에 있는 '날짜' 컬럼을 인덱스에서 제거하여 컬럼으로 변환
    read_df.reset_index(drop=False, inplace=True)
    
    #해당 데이터프레임에 Ticker 값을 'CODE' 라는 컬럼으로 추가
    #read_df['CODE'] = ticker #이렇게 컬럼을 추가할 수도 있지만
    #insert() 함수를 사용하면 원하는 위치에 컬럼 추가 가능. 첫 번째 컬럼이 0, 두 번째 컬럼은 1
    #두 번째 컬럼에 CODE 값을 넣고 싶으니까 insert(1, 'CODE', ticker) ㅎㅎ 
    read_df.insert(1,'CODE', ticker)
    print(read_df.head(10))
    
    # <1단계 데이터 가공> 사전에 정의해 둔 data_processing() 함수를 사용 
    price_df, ym_keys = data_preprocessing(read_df, ticker, base_date)
    
    # 가공한 데이터를 미리 생성해 둔 stock_df에 붙이기 수행
    stock_df = stock_df.append(price_df, sort=False)
    
    # <루프 내에서 마지막으로 월별 상대모멘텀 계산을 위한 1개월간 수익률 계산>
    for ym in ym_keys:
        m_ret = price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],'종가'] \
        / price_df.loc[price_df[price_df['STD_YM'] == ym].index[0],'종가'] 
        price_df.loc[price_df['STD_YM'] == ym, ['1M_RET']] = m_ret
        month_last_df = month_last_df.append(price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],\
                                                            ['날짜','CODE','1M_RET']])    

#추출된 데이터프레임 예의상 확인 ^^    
print(stock_df.tail(50))
print(month_last_df.tail(50))

### 2단계 : 데이터 가공 ###
#1) 직전 1개월 수익률을 기준으로 상대 모멘텀 지수를 계산하는 과정 수행해 보자
#간단하게 pivot() 함수를 사용해서 수행
month_ret_df = month_last_df.pivot('날짜', 'CODE', '1M_RET').copy()
print(month_ret_df.head(10))

#2) 투자종목으로 선택한 종목을 추출하기 위해 Rank순으로 정렬
month_ret_df = month_ret_df.rank(axis=1, ascending=False, method="max", pct=True)

#3) 상위 40%에 드는 종목들만 신호 목록에 포함시키고 나머지는 np.nan 값을 채운ㄷ.
#   이 작업은 월말에 매수 대기 신호가 발생한 종목과 그렇지 않은 종모글 구분하기 위한 것
#   상대 모멘텀 지수를 충족하는 종목이 어디에서 신호가 발생하는지를 피벗팅 테이블로 보기 명확하다.
month_ret_df = month_ret_df.where(month_ret_df < 0.4, np.nan)
month_ret_df.fillna(0,inplace=True)
month_ret_df[month_ret_df != 0] = 1
stock_codes = list(stock_df['CODE'].unique())

#   상위 10개 날짜에 잡힌 신호 데이터를 확인해 보자 !!!
print(month_ret_df.head(10))

### 3단계 : 포지션 처리 (signal positioning) ###
#1) 포지션 처리 : 각 월말에 거래 신호가 나타난 종목을 대상으로 포지션 처리 수행
#   위 코드 결과로 나타난 데이터 프레임에서 월 말일자별로 1로 표시된 종목과 0으로 표시된 종목 확인
#   0에서 1로 변경되면 제로 포지션에서 매수 포지션으로 변경
#   1에서 0으로 변경되면 청산해야 하는 것
#   1에서 1은 계속 보유상태를 유지한다는 의미임, 따라서 아래와 같이 처리하자
sig_dict = dict() #새로운 딕셔너리 생서. 키와 값으로 이루어진 데이터 형

#거래장부를 생성하는 함수를 미리 정의
def create_trade_book(sample, sample_codes):
    book = pd.DataFrame()
    book = sample[sample_codes].copy()
    book['STD_YM'] = book.index.map(lambda x : datetime.strftime(x, '%Y-%m'))
    
    #거래장부를 만들 때 피벗으로 생성된 거래신호 데이터프레임을 편집
    # 날짜와 종목으로 만들어진 데이터프레임 옆에 포지션을 나타내는 p와 수익률 r을 붙여서
    # p + 종목코드, r + 종목 코드 형태로 컬럼을 추가해서 거래장부를 만들어 놓자
    for c in sample_codes:
        book['p '+c] = ''
        book['r '+c] = ''
    return book

#2) 각 월말에 거래신호가 나타난 종목을 추출하는 부분
for date in month_ret_df.index:
    
    #신호가 포착된 종목 코드만 읽어 오기
    ticker_list = list(month_ret_df.loc[date,month_ret_df.loc[date,:] >= 1.0].index)
    # 날짜별 종목코드 저장
    sig_dict[date] = ticker_list
    
stock_c_matrix = stock_df.pivot('날짜', 'CODE', '종가').copy()
book = create_trade_book(stock_c_matrix, list(stock_df['CODE'].unique())) 

print(book.head(10))

#3) 거래실행 부분
#   - 포지션을 잡았으니 아제 거래를 실행하는 단계의 코드를 구현해 보자

#상대 모멘텀 드레이딩
#  먼저 거래를 실행하는 부분을 함수로 정의해 놓자
#   상대모멘텀 tradings 함수
def tradings(book, s_codes):
    std_ym = ''
    buy_phase = False
    # 종목코드별 순회
    for s in s_codes : 
        print(s)
        # 종목코드 인덱스 순회
        for i in book.index:
            # 해당 종목코드 포지션을 잡아준다. 
            if book.loc[i,'p '+s] == '' and book.shift(1).loc[i,'p '+s] == 'ready ' + s:
                std_ym = book.loc[i,'STD_YM']
                buy_phase = True
            # 해당 종목코드에서 신호가 잡혀있으면 매수상태를 유지한다.
            if book.loc[i,'p '+s] == '' and book.loc[i,'STD_YM'] == std_ym and buy_phase == True : 
                book.loc[i,'p '+s] = 'buy ' + s
            
            if book.loc[i,'p '+ s] == '' :
                std_ym = None
                buy_phase = False
    return book

# 3-1. 포지셔닝
for date,values in sig_dict.items():
    for stock in values:
        book.loc[date,'p '+ stock] = 'ready ' + stock
        
# 3-2. tradings
book = tradings(book, stock_codes)
print(book.head())

### 4단계 : 전략수익률 계산해 보기 ###
#1) 전략수익률 계산 함수 정의
#   마지막 단계의 첫 번째 단계는 거래장부(book)에 적혀 있는 거래 수익률을 계산하는 것
#   이를 위해 별도의 함수를 먼저 만들어 놓자 (multi_returns)
def multi_returns(book, s_codes):
    # 손익 계산
    rtn = 1.0
    buy_dict = {}
    num = len(s_codes)
    sell_dict = {}
    
    for i in book.index:
        for s in s_codes:
            if book.loc[i, 'p ' + s] == 'buy '+ s and \
            book.shift(1).loc[i, 'p '+s] == 'ready '+s and \
            book.shift(2).loc[i, 'p '+s] == '' :     # long 진입
                buy_dict[s] = book.loc[i, s]
#                 print('진입일 : ',i, '종목코드 : ',s ,' long 진입가격 : ', buy_dict[s])
            elif book.loc[i, 'p '+ s] == '' and book.shift(1).loc[i, 'p '+s] == 'buy '+ s:     # long 청산
                sell_dict[s] = book.loc[i, s]
                # 손익 계산
                rtn = (sell_dict[s] / buy_dict[s]) -1
                book.loc[i, 'r '+s] = rtn
                print('개별 청산일 : ',i,' 종목코드 : ', s , 'long 진입가격 : ', buy_dict[s], ' |  long 청산가격 : ',\
                      sell_dict[s],' | return:', round(rtn * 100, 2),'%') # 수익률 계산.
            if book.loc[i, 'p '+ s] == '':     # zero position || long 청산.
                buy_dict[s] = 0.0
                sell_dict[s] = 0.0

    acc_rtn = 1.0        
    for i in book.index:
        rtn  = 0.0
        count = 0
        for s in s_codes:
            if book.loc[i, 'p '+ s] == '' and book.shift(1).loc[i,'p '+ s] == 'buy '+ s: 
                # 청산 수익률계산.
                count += 1
                rtn += book.loc[i, 'r '+s]
        if (rtn != 0.0) & (count != 0) :
            acc_rtn *= (rtn /count )  + 1
            print('누적 청산일 : ',i,'청산 종목수 : ',count, \
                  '청산 수익률 : ',round((rtn /count),4),'누적 수익률 : ' ,round(acc_rtn, 4)) # 수익률 계산.
        book.loc[i,'acc_rtn'] = acc_rtn
    print ('누적 수익률 :', round(acc_rtn, 4))

#  변경된 거래장부 일부 살펴보기
print('변경된 거래장부 중 일부 내용')
book.loc['2021-06-01':'2021-09-30',['005930','p 005930','r 005930']]

#2) 전략수익률 계산 실행
#  상대 모멘텀의 전략 수익률은 다수 종목을 대상으로 하므로 단일 종모그로 계산했던
#  절대 모멘텀 수익률을 계산하는 것과 다르다.
#  처음 포지션에 진입할 때의 패턴과 매도할 때의 포지션 패턴을 분석해 보면
#  처음 포지션 진입의 경우 매수 당일 'buy' 신호, 매수 전날 'ready' 신호
#  매수 이틀 전날에는 보유한 포지션이 없는지 확인한다.
#  또한 청산할 때 매도 신호가 있는지 확인한다.
#  다음으로 누적수익률 계산 시 청산할 때 수익이 실현되므로 청산 종목 수로 실현 수익률을 나눠
#  누적 수익률에 반영하면 된다.

#작성된 함수를 실행하면 전략 실행까지 완성된다.
#  이로써 상태 모멘텀 전략을 작성/검토 완료
multi_returns(book, stock_codes)





   