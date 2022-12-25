![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Quant%20Trade&fontSize=90&animation=fadeIn&fontAlignY=38&desc=Discover%20Algorithm%20for%20quant%20trading&descAlignY=51&descAlign=62)
<p align='center'> Discover Quant Algorithm with Python !!! </p>

## 1.주의사항
- V1, V2는 1시간, mini는 하루 간격으로 업데이트됩니다.
- 만약 로드된 후 애니메이션을 다시 보고싶으시면 `ctrl + shift + R` 을 눌러서 강력 새로고침을 하시면 다시 보실 수 있습니다!


## 2. 개발환경 구축

### 2.1.1 아나콘다 다운로드 및 설치
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


### requirements.txt를 이용할 경우

```sh
pip install -r requirements.txt
python manage.py runserver # 서버 실행
```
