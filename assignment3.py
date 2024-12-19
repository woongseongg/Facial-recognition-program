"""

    얼굴 인식을 활용한 도어락 프로그램

    < 기능 작성 순서>
    1. 여러 환경에서 찍은 한 사용자 얼굴 이미지를 학습
    2. 이미지를 통한 얼굴 인식
    3. 등록된 사용자일 경우 통과

    < 환경 가정 >
    1. 사용자가 카메라에 얼굴을 비추면 사진을 촬영
    2. 촬영된 얼굴 이미지가 본 프로그램에 전달됨
    3. 본 프로그램의 연산 과정을 거쳐 반환된 값을 통해 이후의 도어락 제어가 실행

    < 학습 단위>
    작성자 본인의 얼굴 사진 약 20장 이용
    충분한 양의 사진을 이용하기 어려워 정확도를 높이기 위해 이미지를 얼굴 영역만으로 특정

"""

import os                   # 파일 경로 및 폴더 작업을 위한 라이브러리
import face_recognition     # 얼굴 이미지에서 벡터를 추출하기 위한 라이브러리
import numpy as np          # 데이터 저장 및 벡터 연산을 위한 라이브러리


"""

    사용자 얼굴 데이터 학습 단계 
    
        - user_name에 따른 이미지에서 특징 벡터 추출
        - 추출한 벡터를 이미지 단위로 Numpy 배열로 저장
    
"""

def get_user_v_data_path(user_name):
    """
        : 사용자 이름에 기반한 얼굴 벡터 데이터 파일 경로를 반환

        :param user_name: 사용자 이름
        :return: 얼굴 벡터 데이터 파일 경로 (str)
    """
    # 벡터 저장 파일 경로 반환
    return f"database/{user_name}_vector_data.npy"


def vector_extraction(user_name):
    """
        : images/user_faces 하위의 사용자 이름에 해당하는 폴더에서 모든 이미지 읽어옴
        : 각 이미지를 읽어 얼굴 영역을 검출하고, 딥러닝 모델로 얼굴 특징 벡터 생성
        : 추출된 벡터를 Numpy 배열로 변환해 로컬 파일에 저장

        :param user_name: 저장된 이미지 파일을 읽기 위해 필요한 사용자 이름
        :return:

    """
    # 사용자 폴더 경로 설정
    folder_path = f"images/user_faces/{user_name}/"
    if not os.path.exists(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return

    # 얼굴 이미지에서 추출한 특징 벡터를 저장하기 위한 리스트
    user_faces_to_vector = []

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

            # 얼굴 검출
            if len(face_encodings) > 0:
                # 첫 번째 얼굴 이미지 벡터를 리스트에 추가
                ## face_encodings[0] : 혹시 두 사람 이상 감지 되었을 때를 위해 첫 번째 얼굴만을 저장
                user_faces_to_vector.append(face_encodings[0])
            else:
                # 실패
                print(f'얼굴을 찾을 수 없습니다: {file_path}')

    # 얼굴 이미지에서 추출한 벡터가 저장되었는지 확인
    if user_faces_to_vector:
        # 벡터 데이터 저장 경로 (없으면) 생성
        ## exist_ok=True : 디렉토리가 이미 있으면 오류를 발생하는 os.makedirs가 정상적으로 넘어가도록 설정
        os.makedirs("database", exist_ok=True)

        # 경로 생성 함수 호출
        save_path = get_user_v_data_path(user_name)

        # np.save()로 추출된 얼굴 벡터 데이터를 Numpy 파일로 저장
        ## 2차원 Numpy 배열 형태로 저장
        ### [ 0.123, 0.456, ..., 0.789(128번째)],
        ### [ ..., ..., ..., ..., ..., ..., ...], ...
        np.save(save_path, user_faces_to_vector)

        # 저장 완료 메세지 출력
        print(f'{user_name}님의 얼굴 데이터가 저장되었습니다!')
    else:
        print(f'{user_name}의 얼굴 데이터 정상적으로 저장되지 않았습니다.')

# 테스트 실행
vector_extraction('wooseong')



def vector_checking(user_name):
    """
        : 저장된 사용자 얼굴 이미지의 벡터 데이터를 출력하여 확인

        :param user_name: 사용자 이름
        :return: 얼굴 벡터 데이터 파일 경로 (str)
    """
    # 사용자 얼굴 벡터 데이터가 저장된 파일 경로 생성
    file_path = get_user_v_data_path(user_name)

    # 데이터 파일이 존재하는지 확인
    if os.path.exists(file_path):
        # Numpy를 사용해 지정된 데이터 로드
        user_faces_to_vector = np.load(file_path)

        # 데이터 크기와 내용 출력
        print(f'{user_name}님의 얼굴 벡터 데이터: {len(user_faces_to_vector)}개의 벡터')
        print(user_faces_to_vector)
    else:
        print(f'{user_name}님의 데이터가 존재하지 않습니다.')

# 테스트 실행
vector_checking('wooseong')






"""
    2.
"""


"""
    3.
"""


"""
    4.
"""


"""
    [ 실시간 얼굴 인식 단계 ]
    5.
"""


"""
    6.
"""


"""
    7.
"""


"""
    8.
"""


"""
    9.
"""