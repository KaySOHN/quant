import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from pykrx import stock
from pykrx import bond
import exchange_calendars as ecals

import pandas as pd
from datetime import datetime, timedelta
from pandas import Series, DataFrame
import sqlite3
from pykrx import stock
from pykrx import bond
import exchange_calendars as ecals

#시작일, 종료일 설정
str = "20150101"
end = "20151231"
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

        #이 프로그램은 특정 지정일로부터 250거래일 전의 가격변동 내역을 추출하는 것
        #작업일 설정
        baseDate = v_TRDate
        fromDate = ""
        v_cnt = 0

        #작업일 datetime으로 변환
        v_Date = baseDate
        cntDate = datetime.strptime(baseDate, "%Y%m%d")

        #250일 전 거래일을 구하기...중간에 휴장일이 있을 수 있으니 휴장을 빼고 더하기
        while v_cnt <= 250 : #while...Loop를 사용하여 250일 개장일을 찾을 때까지 루핑
            v_Date = cntDate.strftime("%Y%m%d") #날짜 값을 다시 캐릭터 값으로
            v_Chk = XKRX.is_session(v_Date)
            
            if v_Chk == False:
                pass
            else:
                v_cnt = v_cnt + 1 
            
            #루프를 돌기 위해 날짜를 하루 빼주기
            cntDate -= timedelta(days=1)

        #최종적으로 구한 250 거래일 전의 날짜를 리턴
        fromDate = v_Date
        print(baseDate, "와 250거래일 떨어진 날짜는 : ", fromDate)    

        #모든 종목의 정해진 두 날짜 사이의 가격변동 추이 조회
        con = sqlite3.connect("c:/stock/pykrx.db")
        df = stock.get_market_price_change(fromDate, baseDate, market="ALL")

        #만들어진 DataFrame에 baseDate, fromDate 컬럼을 추가
        df.reset_index(drop=False, inplace=True)
            
        #나중을 위해 TRDate를 문자형태로 변경
        df['baseDate'] = baseDate
        df['fromDate'] = fromDate

        #생성된 DataFrame에서 등락률이 30이하인 데이터는 삭제하자 !!!
        dropCond = df[(df['등락률'] <= 30)].index
        df_pr = df.drop(dropCond)
        
        #최종 DataFrame을 DB에 저장    
        df_pr.to_sql('price250', con, if_exists = 'append')
            
    #하루 더하기
    strDate += timedelta(days=1)


