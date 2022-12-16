import pandas as pd
from datetime import datetime, timedelta
from pandas import Series, DataFrame
import sqlite3
from pykrx import stock
from pykrx import bond
import exchange_calendars as ecals

#시작일, 종료일 설정
str = "20220101"
end = "20221125"
XKRX = ecals.get_calendar("XKRX") # 한국 코드를 변수로 설정

#시작일, 종료일 datetime으로 변환
strDate = datetime.strptime(str, "%Y%m%d")
endDate = datetime.strptime(end, "%Y%m%d")

#종료일까지 반복
while strDate <= endDate:

    #거래일 체크를 위해서 현재 날짜를 다시 텍스트로 변환
    v_TRDate = strDate.strftime("%Y%m%d")
    
    #제일 먼저 선택된 날짜의 요일을 구한다.
    v_Chk = XKRX.is_session(v_TRDate)
        
    #토요일/일요일은 작업에서 배제하기 위해 아래 if를 추가
    if v_Chk == False:
        print(v_TRDate, "선택하신 날짜는 휴장일입니다 !!!")
        pass

    else:
        
        print(v_TRDate)
        #일자별 전체종목 시세조회
        df = stock.get_market_fundamental(v_TRDate, market="ALL")

        #만들어진 DataFrame에 TRDate 컬럼을 추가
        df.reset_index(drop=False, inplace=True)
        df['TRDate'] = v_TRDate
        
        #pykrx DB에 접속 및 ohlcv 테이블 데이터 추가
        con = sqlite3.connect("c:/stock/pykrx.db")
        df.to_sql('fund', con, if_exists = 'append')
    
    #하루 더하기
    strDate += timedelta(days=1)
