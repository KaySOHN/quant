![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=capsule%20render&fontSize=90&animation=fadeIn&fontAlignY=38&desc=Decorate%20GitHub%20Profile%20or%20any%20Repo%20like%20me!&descAlignY=51&descAlign=62)
<p align='center'> Discover Quant and Algorithm with me! </p>
<p align='center'>
  <a href="https://github.com/kyechan99/capsule-render/labels/Idea">
    <img src="https://img.shields.io/badge/IDEA%20ISSUE%20-%23F7DF1E.svg?&style=for-the-badge&&logoColor=white"/>
  </a>
  <a href="#demo">
    <img src="https://img.shields.io/badge/DEMO%20-%234FC08D.svg?&style=for-the-badge&&logoColor=white"/>
  </a>
</p>

## 1.주의사항
- V1, V2는 1시간, mini는 하루 간격으로 업데이트됩니다.
- 만약 로드된 후 애니메이션을 다시 보고싶으시면 `ctrl + shift + R` 을 눌러서 강력 새로고침을 하시면 다시 보실 수 있습니다!


## 2. 개발환경 구축

### 2.1.1 아나콘다 다운로드 및 설치
- 파이썬 인터프리터로 아나콘다 배포판을 사용
- 아래 링크에서 다운로드 및 설치
![image](https://user-images.githubusercontent.com/120305891/209038337-f96f0f8b-8a52-4498-901a-d81ff98e0957.png)
- 단, 증퀀사 API를 사용하기 위해서는 32-Bit를 설치하던가,
  64-Bit 버전을 설치한 후 가상환경 설치 시 32-Bit 버전으로 설치해야 함.
- https://www.anaconda.com/products/individual
```sh
poetry install             # 의존성 설치
poetry shell               # 가상환경에 진입
python manage.py runserver # 서버 실행
```

### requirements.txt를 이용할 경우

```sh
pip install -r requirements.txt
python manage.py runserver # 서버 실행
```

## Mazassumnida v.1.0
