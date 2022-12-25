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
(base) c:\Andaconda3>conda create -n quant python=3.8 #quant 라는 이름의 가상환경 생성
(base) c:\Andaconda3>conda env list  #생성된 가상환경 리스트를 점검
(base) c:\Andaconda3>conda activate quant  #quant라는 이름의 가상환경에 진입
(quant) c:\Andaconda3>

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

### 2.4 SQLite3 ODBC 설치 및 연동

#### 2.4.1 SQLite3 ODBC Driver 설치
- 아래 링크에서 해당 Driver를 다운로드
- http://www.ch-werner.de/sqliteodbc/
- sqliteodbc.exe, sqliteodbc_w64.exe 다운로드 후 설치하면 됨

#### 2.4.2 생성한 DB를 ODBC 드라이버에 연결시키기
- 제어판-관리도구 실행
- ODBC Data Sources 실행 (64bit 예시)

- SQLite3 ODBC Driver 선택

- ODBC 드라이버에 연동 할 Data Source 파일 이름 지정 및 옵션 선택 
![image](https://user-images.githubusercontent.com/120305891/209465198-f7a0cf56-0090-48f3-9a21-f9f3aae91bd6.png)

- 생성한 데이터 소스를 확인하고 OK
![Uploading image.png…]()

#### 2.4.3 파워빌더에서 ODBC를 이용해 sQLite3 접속하기 
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



