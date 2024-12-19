import os
import numpy as np
import cv2 as cv
import face_recognition
from config import IMAGE_DIR, VECTOR_DIR, get_vector_data_path
from scipy.signal import dfreqresp

"""
    : data/user_faces 하위의 사용자 이름에 해당하는 폴더에서 모든 이미지 읽어옴
    : 각 이미지를 읽어 얼굴 영역을 검출하고, 딥러닝 모델로 얼굴 특징 벡터 생성
    : 추출된 벡터를 Numpy 배열로 변환해 로컬 파일에 저장

    :param user_name: 사용자 이름
    :return:

"""


def vector_extraction(user_name):
    # 사용자 폴더 경로 설정
    folder_path = os.path.join(IMAGE_DIR, user_name)
    if not os.path.exists(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return

    # 얼굴 이미지에서 추출한 특징 벡터를 저장하기 위한 리스트
    face_to_vectors = []

    # listdir()로 지정된 폴더 내 모든 이미지 파일 읽기
    for file_name in os.listdir(folder_path):
        # 이미지 파일만
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 파일 경로 생성 (폴더 경로와 파일 이름 결합)
            file_path = os.path.join(folder_path, file_name)
            print(f'이미지 로드 중: {file_path}')

            # 이미지 파일을 읽어서 로드
            ## shape은 (h, w, 3)
            image = face_recognition.load_image_file(file_path)

            # 얼굴 이미지에서 128차원 특징 벡터 추출
            ## 128차원 벡터는 동일한 사람에 대해서는 거의 동일한 값을 보이지만,
            ## 서로 다른 사람 간에는 큰 차이를 보임
            face_encodings = face_recognition.face_encodings(image)

            # 디버깅용 메세지
            #print(face_encodings)

            # 얼굴 검출
            if len(face_encodings) > 0:
                # 첫 번째 얼굴 이미지 벡터를 리스트에 추가
                ## face_encodings[0] : 혹시 두 사람 이상 감지 되었을 때를 위해 첫 번째 얼굴만을 저장
                face_to_vectors.append(face_encodings[0])
            else:
                # CNN 모델로 얼굴 감지
                ## 얼굴 위치 좌표 리스트를 리턴
                face_locations = face_recognition.face_locations(image, model="cnn")
                # 디버깅용 출력
                print('감지된 위치', face_locations)

                if not face_locations:
                    # 실패
                    print(f'얼굴을 감지하지 못했습니다.')
                else:
                    #print(f'감지된 얼굴 개수: {len(face_locations)}')
                    # face_locations 리스트의 각 (top, right, bottom, left) 좌표 튜플을 순회
                    #for (top, right, bottom, left) in face_locations:
                        # 순회하면서 이미지에 얼굴 위치를 특정해 사각형을 그림
                        ## image : Numpy 배열 이미지
                        ## (left, top) : 사각형의 왼쪽 상단 좌표
                        ## (right, bottom) : 사각형의 오른쪽 하단 좌표
                        ## (0, 255, 0) : 사각형의 색상 (BGR 형식 튜플), (0, 255, 0)은 녹색
                        ## 2 : 사각형의 선 두께 (픽셀)
                        #cv.retangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

                    for face_location in face_locations:
                        # 특징 벡터 추출
                        face_encodings = extract_vector_from_rectangle(image, face_location)
                    # 추출한 128차원 벡터를 배열에 저장
                    face_to_vectors.append(face_encodings[0])

    # vector_store(face_to_vectors, user_name)

    face_vectors_array = np.array(face_to_vectors)  # 리스트 -> Numpy 배열
    vector_file_path = os.path.join(VECTOR_DIR, f"vector_data_{user_name}.npy")  # 저장 경로
    np.save(vector_file_path, face_vectors_array)


def vector_store(vector_list, user_name):
    """
    사용자 얼굴 벡터 데이터를 저장

    :param vector_list: 이미지 별로 추출한 특징 벡터를 저장한 리스트
    :param user_name : 사용자 이름
    :return:
    """
    # 얼굴 이미지에서 추출한 벡터 저장
    if vector_list:
        # 벡터 데이터 저장 경로 (없으면) 생성
        ## exist_ok=True : 디렉토리가 이미 있으면 오류를 발생하는 os.makedirs가 정상적으로 넘어가도록 설정
        os.makedirs(VECTOR_DIR, exist_ok=True)

        # 경로 생성 함수 호출
        save_path = get_vector_data_path(user_name)
        print('save_path: ', save_path)

        # np.save()로 추출된 얼굴 벡터 데이터를 Numpy 파일로 저장
        ## 2차원 Numpy 배열 형태로 저장
        ### [ 0.123, 0.456, ..., 0.789(128번째)],
        ### [ ..., ..., ..., ..., ..., ..., ...], ...
        np.save(save_path, vector_list)

        # 저장 완료 메세지 출력
        print(f'{user_name}님의 얼굴 데이터가 저장되었습니다!')
    else:
        print(f'{user_name}의 얼굴 데이터 정상적으로 저장되지 않았습니다.')



def extract_vector_from_rectangle(img, rectangle):
    """
    특정 사각형 영역에서 128차원 특징 벡터를 추출

    :param img: Numpy 배열의 입력 이미지
    :param rectangle: 사각형 좌표 (top, right, bottom, left)
    :return: 128차원 Numpy 배열 형태의 벡터
    """
    if not rectangle or len(rectangle) != 4:
        raise ValueError('유효한 사각형 좌표 튜플 (top, right, bottom, left)가 필요합니다.')
    # 디버깅용 출력
    print('rectangle:', rectangle)

    # 입력 이미지를 RGB 형식으로 변환
    ## 흑백이면 img.shape == 3
    ## 여기서는 3 차원 배열이며, 채널 수가 3개(BRG 또는 RGB)일 때
    if len(img.shape) == 3 and img.shape[2] == 3:
        rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:   # 흑백 이미지 등 3채널이 아닌 경우에는 형식에 관계 없으므로 그대로 대입
        rgb_image = img

    # 사각형의 얼굴 영역 추출
    top, right, bottom, left = rectangle
    # 얼굴 영역에서 마스크 부분은 제외하고 사용
    bottom = 2 * (top + bottom) // 5
    cropped_image = rgb_image[top:bottom, left:right]

    # 얼굴 벡터 추출 후 저장
    face_encodings = face_recognition.face_encodings(cropped_image)
    if face_encodings:
        return face_encodings[0]
    else:
        print('사각형 내에서 얼굴을 감지하지 못했습니다.')
        return None