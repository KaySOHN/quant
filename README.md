![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Quant%20Trade&fontSize=90&animation=fadeIn&fontAlignY=38&desc=Discover%20Algorithm%20for%20quant%20trading&descAlignY=51&descAlign=62)
<p align='center'> Discover Quant Algorithm with Python !!! </p>

## 1.주의사항
- V1, V2는 1시간, mini는 하루 간격으로 업데이트됩니다.
- 만약 로드된 후 애니메이션을 다시 보고싶으시면 `ctrl + shift + R` 을 눌러서 강력 새로고침을 하시면 다시 보실 수 있습니다!


## 2. 개발환경 구축

### 2.1. 아나콘다 다운로드 및 설치
- 파이썬 인터프리터로 아나콘다 배포판을 사용
- 아래 링크에서 다운로드 및 설치
![image](https://user-images.githubusercontent.com/120305891/209038337-f96f0f8b-8a52-4498-901a-d81ff98e0957.png)
- 단, 증권사 API를 사용하기 위해서는 32-Bit를 설치하던가,
  64-Bit 버전을 설치한 후 가상환경 설치 시 32-Bit 버전으로 설치해야 함.
- 일단 우리는 64비트 아나콘다를 설치한 후 32비트 환경을 추가로 셋팅하지.
- https://www.anaconda.com/products/individual
- 아나콘다 64비트 다운로드 후 설치 후 가상환경 생성
- Anaconda Prompt를 실행하여 다음과 같이 입력한다.
```sh
(base) c:\Andaconda3>set CONDA_FORCE_32BIT = 1 # 강제로 32Bit로 셋팅
(base) c:\Andaconda3>conda create -n quant python=3.8.5 #quant 라는 이름의 가상환경 생성
- pykiwoom을 위해서는 python 3.8.5가 필요
(base) c:\Andaconda3>conda env list  #생성된 가상환경 리스트를 점검
(base) c:\Andaconda3>conda activate quant  #quant라는 이름의 가상환경에 진입
- 가상환경의 운영 Bit 및 파이썬 버전 확인
(quant) c:\Andaconda3>conda info
- 가상환경의 삭제
(quant) c:\Andaconda3>conda env remove -n quant


```

### 2.2 Visual Studio Code 설치 및 환경 설정
- VSCode는 마이크로소프트가 개발한 무료 소스 코드 편집기
- 아래 링크에서 Windows 64비트용을 다운로드 및 설치
- http://code.visualstudio.com
- 설치 과정 중 아래 그림에서 모든 항목 선택 후 "다음" 버튼
![image](https://user-images.githubusercontent.com/120305891/209459032-0d06a400-f97c-46b7-acb2-eef949db0304.png)

### 2.3 Visual Studio Code 에서 Python 설치
- 확장(단축키 : Ctrl + Shift + X)을 실행하여 Python을 검색 : Microsfot가 만든 Python을 설치하면 
- Python extension for Visual Studio Code 설치
- Python 설치 시 : InteliSense, Liniting, 디버깅, 코드 탐색, Jupyter Notebook 지원 등의 기능 지원 확인
- Python for VSCode 설치 : 파이썬 언어팩, 구문강조 스니펫 등 기능 지원
- Python Extension Pack 설치 : 멀트쓰레드 디버깅 지원 IntelliCode 지원 가능
![image](https://user-images.githubusercontent.com/120305891/209464618-d560eb0e-2a49-489e-bc3b-c194612e3d73.png)

- 터미널 변경하기 : 아나콘다는 명령 프롬프트와 잘 동작하므로 기본 터미널을 통해 명령 프롬프트로 변경 
 
### 2.4 PowerBuilder 10.0, 10.2, 10.5 설치 및 패치
- 먼저 PowerBuilder 10.0을 Install
- 다음, PowerBuilder 10.2 패치 : PowerBuilder10_Patch 가 바로 PowerBuilder 10.2 패치임
- 다음, PowerBuilder 10.5 설치 : Sybase PowerBuilder Enterprise 10.5를 가상 드라이브 마운트 후 설치 
- 마지막으로 PB105_5034 ~ PB105_7599 까지 순서대로 패치하면 설치 완료

### 2.5 SQLite3 ODBC 설치 및 연동

#### 2.5.1 SQLite3 ODBC Driver 설치
- 아래 링크에서 해당 Driver를 다운로드
- http://www.ch-werner.de/sqliteodbc/
- sqliteodbc.exe, sqliteodbc_w64.exe 다운로드 후 설치하면 됨

#### 2.5.2 생성한 DB를 ODBC 드라이버에 연결시키기
- 제어판-관리도구 실행
- ODBC Data Sources 실행 (64bit 예시)

- SQLite3 ODBC Driver 선택

- ODBC 드라이버에 연동 할 Data Source 파일 이름 지정 및 옵션 선택 
![image](https://user-images.githubusercontent.com/120305891/209465198-f7a0cf56-0090-48f3-9a21-f9f3aae91bd6.png)

- 생성한 데이터 소스를 확인하고 OK
![image](https://user-images.githubusercontent.com/120305891/209465500-7661784c-35dc-426e-af58-b4f60d4f4fff.png)

#### 2.5.3 파워빌더에서 ODBC를 이용해 sQLite3 접속하기 
- 처음 Application의 Open Event-PowerScript에서 아래와 연결
```sh
String ls_dbSQLite, ls_db
 
ls_dbSQLite = "C:\sqlite.db" //sqlite.sqlite
ls_db = "PBSQLITE"
 
Transaction ltran_conn
ltran_conn = Create Transaction
Disconnect Using ltran_conn ;
 
RegistrySet("HKEY_CURRENT_USER\SOFTWARE\ODBC\ODBC.INI\"+ls_db+"","AutoStop",RegString!,"yes")
RegistrySet("HKEY_CURRENT_USER\SOFTWARE\ODBC\ODBC.INI\"+ls_db+"" ,"Database",RegString!,ls_dbSQLite)
RegistrySet("HKEY_CURRENT_USER\SOFTWARE\ODBC\ODBC.INI\"+ls_db+"" ,"Driver",RegString!,"sqlite3odbc.dll")
 
// Using ODBC Connect To SQLite 
ltran_conn.DBMS = "ODBC"
ltran_conn.AutoCommit = False
ltran_conn.DBParm =  "ConnectString='DSN=" + ls_db + "'"
 
Connect Using ltran_conn ;
If ltran_conn.SQLCode = -1 Then
	MessageBox('Warning','Connect Database Error' + ltran_conn.SQLErrText)
Else
	MessageBox('Warning',"Connect Success!")
End If
 
Disconnect Using ltran_conn ;
```
 
- 여기까지 하면 개발환경 준비 끝^^

### 2.6 금융 관련 패키지 설치하기
-  Python에서 금융/주식 관련 프로그램을 실행하기 위해서는 관련 패키지 설치가 필수적이다.
-  아래는 관련 패키지를 미리 생성한 quant 가상환경에서 설치하는 과정이다.
-  Anaconda Prompt (Anacoda3)를 실행 후 아래와 같이 입력한다.
```sh
(base) c:\Andaconda3>conda activate quant  #quant라는 이름의 가상환경에 진입
(quant) c:\Andaconda3>

#Pandas, pyqt5, matplotlib 패키지 설치
(quant) c:\Andaconda3>pip install pandas pyqt5 matplotlib

#pywin32 모듈 설치 : pip로 설치할 경우 에러가 발생하면 conda로 설치한다.
(quant) c:\Andaconda3>conda install pywin32

#BeautifulSoup 모듈
(quant) c:\Andaconda3>pip install beautifulsoup4

#pandas DataReader 모듈
(quant) c:\Andaconda3>pip install pandas_datareader

#주식시장 개장일/휴장일 관련 모듈 : pandas-market-calendars, trading-calendars,  exchange_calendars 등
(quant) c:\Andaconda3>pip install pandas_market_calendars
(quant) c:\Andaconda3>pip install trading-calendars
(quant) c:\Andaconda3>pip install exchange_calendars

#IPython 모듈
(quant) c:\Andaconda3>pip install ipython

#pykrx 모듈 설치
(quant) c:\Andaconda3>pip install pykrx

#pystock 모듈 설치
(quant) c:\Andaconda3>pip install pystock
(quant) c:\Andaconda3>pip install pystocklib
(quant) c:\Andaconda3>pip install pysrim

#pykiwoom 모듈 설치
(quant) c:\Andaconda3>pip install pykiwoom
(quant) c:\Andaconda3>pip install -U pykiwoom

#backtesting 모듈 설치
(quant) c:\Andaconda3>pip install backtesting
(quant) c:\Andaconda3>pip install backtrader
(quant) c:\Andaconda3>pip install quandl

#증권 데이터 제공 모듈
(quant) c:\Andaconda3>pip3 install quandl

#캔들 챠트 라이브러리
(quant) c:\Andaconda3>pip install mpl_finance
(quant) c:\Andaconda3>pip install mplfinance

#yahoo finance 모듈
(quant) c:\Andaconda3>conda install -c ranaroussi yfinance
(quant) c:\Andaconda3>pip install yfinance --upgrade --no-cache-dir

#TA-Lib 모듈
(quant) c:\Andaconda3>pip install ta
(quant) c:\Andaconda3>pip install pandas_ta
(quant) c:\Andaconda3>pip install TA-Lib

#Excel 모듈
(quant) c:\Andaconda3>pip install openpyxl

#Cufflinks 모듈
(quant) c:\Andaconda3>pip install cufflinks
(quant) c:\Andaconda3>pip install chart_studio

#candle chart 그리기 모듈
(quant) c:\Andaconda3>pip install mplfinance
(quant) c:\Andaconda3>pip install mpl_finance

#케라스(Keras) 모듈
(quant) c:\Andaconda3>pip install keras

#젠심(Gensim) 모듈
(quant) c:\Andaconda3>pip install gensim

#XGBoost 모듈
(quant) c:\Andaconda3>pip install xgboost

#사이킷런(Scikit-learn) 모듈
(quant) c:\Andaconda3>pip install scikit-learn

#텐서플로우(Tensorflow) 모듈 : 64Bit에서만 설치 가능
(base) c:\Andaconda3>pip install tensorflow
#ipython 쉘을 실행하여 텐서플로우가 정상실행되는지 임포트하고 버전을 확인
(base) c:\Andaconda3>ipython
import tensorflow as tf
tf.__version__

### 2.7 Facebokk Prophet 관련 키지 설치하기
#### 2.7.1 Python 버전 확인하기 및 가상환경 활성화
-  python 가상환경 3.8버전에서 가상환경 활성화 (python 3.8에서만 가능)
(quant) c:\Andaconda3>python -V
(quant) c:\Andaconda3>conda activate quant

#### 2.7.2 C++ 컴파일러 설치
(quant) c:\Andaconda3>conda install libpython m2w64-toolchain -c msys2

#### 2.7.3 Prophet 및 fbprophet에 대한 종속 항목들 설치
(quant) c:\Andaconda3>conda install numpy cython -c conda-forge
(quant) c:\Andaconda3>conda install matplotlib scipy pandas -c conda-forge
(quant) c:\Andaconda3>conda install pystan -c conda-forge
(quant) c:\Andaconda3>conda install -c anaconda ephem

#### 2.7.4 시계열 예측에 사용될 라이브러리를 설치
(quant) c:\Andaconda3>pip install scikit-learn
(quant) c:\Andaconda3>pip install auto-arima (pmdarima)
(quant) c:\Andaconda3>pip install fbprophet
(quant) c:\Andaconda3>pip install plotly

#### 2.7.5 Prophet 설치
(quant) c:\Andaconda3>pip install pystan==2.19.1.1 prophet

또는

(quant) c:\Andaconda3>pip install -c conda-forge prophet


```

### 2.8 키움증권 OpenAPI 설치하기 
-  OpenAPI 설치 : 키우증원 OpenAPI 모듈을 다운로드하여 C:\OpenAPI 폴더에 복사
-  KOA Studio 파일을 다운로드 하여 C:\OpenAPI 폴더에 복사하면 됨.

