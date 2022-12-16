from yahoo_fin.stock_info import *
import yahoo_fin.stock_info as si 
from yahoo_fin.stock_info import get_analysts_info
import pandas as pd
from functools import reduce


print(si.get_analysts_info('nflx'))
print(si.get_balance_sheet('nflx'))
print(si.get_cash_flow('nflx'))

mstf_data = si.get_data('msft')
print(mstf_data.head())

print(si.get_data("amzn", start_date = "04/01/2020", end_date = "04/30/2020"))

# get list of 'amzn', 'ba', 'msft', 'aapl', 'goog' tickers
sp = ['amzn', 'ba', 'msft', 'aapl', 'goog']#si.tickers_sp500()
 
# pull data for each stock
price_data = {ticker : si.get_data(ticker, start_date = "04/01/2020", end_date = "04/30/2020") for ticker in sp}
combined = reduce(lambda x,y: x.append(y), price_data.values())

print(combined.head())
print(combined.tail())

#매매일 현재 상승 상위100개종목
print(si.get_day_gainers())
     
#매매일 현재 하락 상위100개종목
print(si.get_day_losers())
     
#거래량 상위정보
print(si.get_day_most_active())
     
#종목의 보유정보
print(si.get_holders('nflx'))
     
#종목의 현재가격
print(si.get_live_price('nflx'))
     
#종목의 호가정보
print(si.get_quote_table('aapl'))

#종목의 통계정보
print(si.get_quote_table('nflx'))

print(si.get_stats_valuation('nflx'))


#다우존스지수 종목의 티커
print(si.tickers_dow())
     
#나스닥종목의 티커
print(si.tickers_nasdaq())
     
#기타 종목의 티커
print(si.tickers_other())
     
#S&P 500 구성종목을 돌려준다( https://en.wikipedia.org/wiki/List_of_S%26P_500_companies )
print(si.tickers_sp500())
     
#암호화 화폐 거래정보
#print(si.get_top_crypto())

#개별종목의 호가정보는 get_quote_table를 사용하여 구한다
quote = si.get_quote_table("aapl")
     
#get_quote_table 메서드는 다음과 같이 여러 정보가 담긴 딕셔너리를 돌려준다.
print(quote)
     
#다음과 같이 키 값(여기선 "PE Ratio (TTM)")을 사용하여 값을 얻을 수 있다.
quote["PE Ratio (TTM)"] # 22.71
     
#PER을 구하는 또 다른 방법은 get_stats_valuation 메서드를 이용하는 것이다.
#이것은 야후파이낸스의 종목 통계섹션의 내용이다.
val = si.get_stats_valuation("aapl")
print(val)

val = val.iloc[:,:2]
val.columns = ["Attribute", "Recent"]
print(val) 

#P/E 비율을 추출하면 다음과 같다
print(float(val[val.Attribute.str.contains("Trailing P/E")].iloc[0,1]))
     
#주가매출액(Price-to-Sales) 비율
print(float(val[val.Attribute.str.contains("Price/Sales")].iloc[0,1]))

#한번에 여러 재무 데이터 구하기
#다우지수내 각 종목의 Price-to-Earnings 와 Price-to-Sales 비율을 구해보도록 하자.
#tickers_dow메서드는 지수를 구성하는 종목의 티커를 돌려준다.


# 각 종목 밸류에이션을 가져온다
# get_stats_valuation()는 데이터프레임을 돌려주는 데, 
# 첫 두개의 컬럼만 가져온다
dow_stats = {}
for ticker in ['amzn', 'ba', 'msft', 'aapl', 'goog']:
    temp = si.get_stats_valuation(ticker)
    temp = temp.iloc[:,:2]
    temp.columns = ["Attribute", "Recent"]
 
    dow_stats[ticker] = temp 

# 가져온 각 종목 데이터를 하나로 합친다 
combined_stats = pd.concat(dow_stats)
combined_stats = combined_stats.reset_index()
 
del combined_stats["level_1"]
 
# 컬럼이름 변경
combined_stats.columns = ["Ticker", "Attribute", "Recent"]
combined_stats

#주가수익률(P/E) 비율
# get P/E ratio for each stock
combined_stats[combined_stats.Attribute.str.contains("Trailing P/E")]
     
#주가매출액비율(Price-to-Sales) 비율
# get P/S ratio for each stock
combined_stats[combined_stats.Attribute.str.contains("Price/Sales")]

#주가순자산(Price / Book) 비율
# get Price-to-Book ratio for each stock
combined_stats[combined_stats.Attribute.str.contains("Price/Book")]

#주가이익성장배율(Price / Earnings-to-Growth) 비율
# get PEG ratio for each stock
combined_stats[combined_stats.Attribute.str.contains("PEG")]

#Forward P/E 비율
# get forward P/E ratio for each stock
combined_stats[combined_stats.Attribute.str.contains("Forward P/E")]
combined_stats

#여러 종목의 기타 통계 구하기
#야후파이낸스 종목 통계섹션에는“Valuation Measures” 테이블이 있는데
#get_stats method를 통해 기타 통계 정보(Return on Equity (ROE),
#Return on Assets, profit margin)를 구할 수 있다.

dow_extra_stats = {}
for ticker in ['amzn', 'ba', 'msft', 'aapl', 'goog']:
    dow_extra_stats[ticker] = si.get_stats(ticker)   
 
combined_extra_stats = pd.concat(dow_extra_stats)
 
combined_extra_stats = combined_extra_stats.reset_index()
 
del combined_extra_stats["level_1"]
 
combined_extra_stats.columns = ["ticker", "Attribute", "Value"]

combined_extra_stats
      
