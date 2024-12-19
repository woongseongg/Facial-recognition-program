# 얼굴 인식 프로그램(Facial recognition program)

## 소개
이 프로젝트는 얼굴 인식 프로그램으로, 사용자의 얼굴 데이터를 처리하고 이를 기반으로 특정 작업을 수행합니다. OpenCV와 같은 컴퓨터 비전 라이브러리를 활용하여 이미지 데이터를 분석하고 머신러닝 모델을 통해 얼굴을 학습 및 검출합니다.


## 주요 기능
1. 얼굴 데이터 캡처: 웹캠을 통해 얼굴 이미지를 캡처합니다.

2. 얼굴 벡터화: 캡처된 이미지를 전처리하여 벡터 형식으로 변환합니다.

3. 모델 학습: 캡처된 데이터를 이용해 얼굴 인식 모델을 학습합니다.

4. 실시간 얼굴 인식: 학습된 모델을 사용하여 실시간으로 얼굴을 검출하고 인식합니다.


## 프로젝트 구조

Facial-recognition-program-main/
│

├── app.py                        # 메인 실행 파일

├── assignment3.py                # 과제 관련 파일

├── config.py                     # 설정 파일

├── models/

│   └── faces_training.py         # 얼굴 학습 관련 코드

├── utils/

│   ├── camera_gui.py             # 카메라 인터페이스 및 GUI 코드

│   ├── preprocessing_of_captured.py # 캡처된 이미지 전처리 코드

│   ├── vector_checking.py        # 벡터 데이터 비교 및 검증

│   └── vector_extraction.py      # 벡터 데이터 추출

├── data/

│   ├── vector_data/              # 얼굴 벡터 데이터 저장 폴더

│   └── captured_images/          # 캡처된 얼굴 이미지 저장 폴더

├── README.md                     # 프로젝트 설명 파일

└── .gitignore                    # Git 무시 파일 설정


## 요구사항
### 소프트웨어 요구사항
1. Python 3.7 이상

2. OpenCV

3. NumPy

4. 기타 라이브러리

### 하드웨어 요구사항
1. 웹캠 또는 카메라

2. 기본적인 컴퓨팅 성능을 갖춘 PC


## 설치 및 실행 방법

1. 프로젝트 클론
```
git clone https://github.com/your-repository-url.git
cd Facial-recognition-program-main
```

필수 라이브러리 설치

pip install -r requirements.txt

프로그램 실행

python app.py

사용 방법

프로그램을 실행하면 GUI를 통해 카메라 인터페이스가 나타납니다.

"캡처" 버튼을 눌러 얼굴 이미지를 저장합니다.

학습 버튼을 통해 얼굴 데이터를 학습합니다.

"인식" 모드를 활성화하여 실시간으로 얼굴을 인식합니다.

기여 방법

이 프로젝트에 기여하려면 다음 단계를 따르세요:

이 저장소를 포크합니다.

새로운 기능을 추가하거나 버그를 수정합니다.

변경 사항을 커밋하고 푸시합니다.

풀 리퀘스트를 생성합니다.
