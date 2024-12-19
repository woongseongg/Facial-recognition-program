import os
import cv2 as cv
import face_recognition


def preprocess_image_and_extract_vector(path="../data/captured_images/"):
    """
    원할한 벡터 추출을 위한 이미지 전처리 작업
    밝기와 대비를 조정하고 벡터를 추출

    :param img_path: 입력 이미지 경로
    :return: 추출된 얼굴 벡터 리스트
    """
    # 이미지 로드
    image = cv.imread(path)
    if image is None:
        print(f'이미지를 로드할 수 없습니다: {path}')
        return []

    # step 1: 밝기와 대비 조정
    bright_image = adjust_brightness_and_contrast(image, alpha=1.5, beta=50)

    # step 2: 히스토그램 균등화 적용
    equalized_image = apply_histogram_equalization(image)

    # step 3: 얼굴 위치 감지
    ## face_recognition() : 얼굴 좌표 (top, right, bottom, left) 형식 튜플 리턴
    ### top: 얼굴 상단 Y좌표
    ### right: 얼굴 우측 X좌표
    ### bottom: 얼굴 하단 Y좌표
    ### left: 얼굴 좌측 X좌표
    face_locations = face_recognition.face_locations(equalized_image)
    if not face_locations:
        print('얼굴을 감지하지 못했습니다.')
        return []

    # step 4: 얼굴 이미지에서 특징 벡터 추출
    ## RGB형식의 equalized_image, 얼굴 위치 좌표 face_locations를 입력받아 128차원 벡터 추출
    face_encodings = face_recognition.face_encodings(equalized_image, face_locations)
    print(f'감지된 얼굴 개수 : {len(face_encodings)}')

    return face_encodings



def adjust_brightness_and_contrast(img, alpha=1.5, beta=50):
    """
    이미지 밝기와 대비를 조정

    :param img: Numpy 배열의 입력 이미지
    :param alpha: 대비 계수 (float)
    :param beta: 밝기 추가 값 (int)
    :return: 조정된 이미지
    """
    # OpenCV의 convertScaleAbs 함수로 밝기와 대비 조정
    adjusted_image = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_image


def apply_histogram_equalization(img):
    """
    히스토그램 균등화 적용을 통해 이미지의 밝기와 대비 개선
    :param img: Numpy 배열의 입력 이미지
    :return: 균등화된 이미지
    """
    # 흑백 이미지로 변환
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 히스토그램 균등화 적용
    equalized_image = cv.equalizeHist(gray_image)

    # 컬러 이미지(BGR 형식)를 위한 3채널 병합
    ## OpenCV의 BGR 형식을 face_recognition 라이브러리의 RGB 형식으로 변환
    return cv.cvtColor(equalized_image, cv.COLOR_GRAY2BGR)