import os

# 이미지 데이터 저장 폴더
IMAGE_DIR = "data/user_faces/"

# 얼굴 벡터 데이터 저장 폴더
VECTOR_DIR = "data/vector_data/"

# 얼굴 캡처 이미지 저장 폴더
CAPTURED_DIR = "data/captured_images/"


# 얼굴 벡터 데이터 저장 파일 경로 반환
def get_vector_data_path(user_name):
    """
        :param user_name: 사용자 이름
        :return: 얼굴 벡터 데이터 파일 경로 (str)
    """
    return os.path.join(VECTOR_DIR, f"vector_data_{user_name}.npy")