import os
import numpy as np
from config import get_vector_data_path

"""
    : 저장된 사용자 얼굴 이미지의 벡터 데이터를 출력하여 확인

    :param user_name: 사용자 이름
    :return: 얼굴 벡터 데이터 파일 경로 (str)
"""

def vector_checking(user_name):
    # 사용자 얼굴 벡터 데이터가 저장된 파일 경로 생성
    file_path = get_vector_data_path(user_name)

    # 데이터 파일이 존재하는지 확인
    if os.path.exists(file_path):
        # Numpy를 사용해 지정된 데이터 로드
        face_to_vectors = np.load(file_path)

        # 데이터 크기와 내용 출력
        print(f'{user_name}님의 얼굴 벡터 데이터: {len(face_to_vectors)}개의 벡터')
        print(face_to_vectors)
    else:
        print(f'{user_name}님의 데이터가 존재하지 않습니다.')