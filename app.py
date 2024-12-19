from utils.vector_extraction import vector_extraction
from utils.vector_checking import vector_checking
from utils.camera_gui import run_CameraApp
from utils.camera_gui import get_img_name
from utils.preprocessing_of_captured import preprocess_image_and_extract_vector
from models.faces_training import measure_similarity
import cv2 as cv

def main():
    user_name = "wooseong"

    """
    저장된 얼굴 이미지에서 특징 벡터를 추출하고 .npy 파일로 저장
    """
    # 얼굴 벡터 추출 및 저장
    # 학습이 잘 된 벡터 저장 파일에 대해서만 수행하도록 주석 처리
    # vector_extraction(user_name)

    # 저장된 얼굴 벡터 확인
    # vector_extraction()과 마찬가지 이유로 주석 처리
    # vector_checking(user_name)



    """
    CameraApp GUI를 실행하고, 얼굴 이미지 캡처 및 벡터 추출
    """
    # 디버깅용 메세지
    # print('카메라 GUI를 실행합니다.')

    # GUI 실행 및 캡처 이미지 반환
    captured_image = run_CameraApp()

    # 캡처된 이미지가 있다면
    if captured_image is not None:
        # 고유한 파일 이름 생성
        unique_img_name = get_img_name(base_name="captured_img", \
                    extension=".jpg", path="data/captured_images/")

        # 캡처 이미지를 고유 이름으로 저장
        cv.imwrite(unique_img_name, captured_image)
    else:
        print('이미지가 정상적으로 저장되지 않았습니다.')



    """
    원할한 벡터 추출을 위해 저장된 이미지 전처리
    """
    # 전처리 및 얼굴 벡터 추출
    face_vectors = preprocess_image_and_extract_vector(unique_img_name)

    # 벡터 확인
    if not face_vectors:
        print('벡터 추출에 실패했습니다.')
        return


    # 디버깅용 메세지
    # 인식된 얼굴 개수 출력
    # print(f'추출된 얼굴 벡터 :')
    # print(face_vectors[0])

    """
    딥러닝 과정 수행
    """
    result = measure_similarity(user_name, face_vectors[0])
    print(result)

if __name__ == "__main__":
    main()