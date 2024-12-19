import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

"""
    1. 추출한 벡터 간 유사도 측정
"""
def measure_similarity(user_name, face_vectors):
    """
    사용자 이름과(user_name) 캡처한 얼굴 벡터 데이터를(face_vectors) 로드해 학습

    데이터 분할 비율:
        데이터가 적으므로(20장) 학습/검증/테스트 비율을 다음과 같이 설정
        학습 데이터: 16개 (80%)
        검증 데이터: 4개 (20%)
        테스트 데이터: 1개 (입력된 사용자 얼굴)

    :param user_name: 사용자 이름
    :param face_vectors: 캡처된 이미지에서 추출한 벡터
    :return: 도어락 열림(1) 또는 닫힘(0) 값 출력
    """
    # 학습 데이터: 16개(80%), 검증 데이터: 4개(20%), 테스트 데이터: 1개(입력된 사용자 얼굴)로 분할
    X_train, X_val, X_test = prepare_data(user_name, face_vectors)

    # 모델 설계
    model = Sequential([
        Input(shape=(128, )),                   # 입력 크기 지정 (128, )
        Dense(units=6, activation='relu'),      # 첫 번째 은닉층
        Dense(units=5, activation='relu'),      # 두 번째 은닉층
        Dense(units=4, activation='relu'),      # 세 번째 은닉층
        Dense(units=1, activation='sigmoid')    # 출력층 (이진 분류)
    ])

    # 모델 컴파일 (손실 함수: 이진 크로스엔트로피, 평가 지표: 정확도)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 구조 요약
    model.summary()

    # 예제 레이블 생성
    ## 학습 데이터 레이블 (1: 같은 사용자)
    y_train = np.ones((X_train.shape[0], 1))
    ## 검증 데이터 레이블
    y_val = np.ones((X_val.shape[0], 1))
    ### y_train : 학습 데이털 레이블 -> 모델이 학습할 때 사용하는 정답 값
    ### y_val : 검증 데이터 레이블 -> 모델이 검증할 때 사용하는 정답 값
    ### X_train[0] == 16
    ### (X_train.shape[0], 1) : 샘플 개수에 따라 1차원 레이블 배열을 생성
    ### 레이블 값이 모두 1이므로, 학습 데이터와 검증 데이터가 같은 사용자로 가정


    # 모델 학습
    ## validation_data=(X_val, y_val) : 검증 데이터와 레이블
    ## batch_size=4 : 학습 데이터를 4개씩 나누어서 한 번에 처리
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=4)

    # 학습 결과 출력 : dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    print(history.history.keys())

    # 학습 손실 값 및 정확도 출력
    print(f"최종 학습 손실: {history.history['loss'][-1]:.4f}")
    print(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
    print(f"최종 학습 정확도: {history.history['accuracy'][-1]:.4f}")
    print(f"최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")

    # 손실 값 그래프
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 테스트 데이터 예측
    test_prediction = model.predict(X_test)  # 입력된 얼굴 벡터 예측

    # 예측 확률 값 출력
    print(f"테스트 데이터 예측값: {test_prediction[0][0]:.4f}")

    # 사용자 확인
    threshold = 0.9  # 임계값 설정
    if test_prediction[0][0] > threshold:
        print(f"{user_name} 사용자 인식")
        return 1
    else:
        print("등록되지 않은 사용자입니다.")
        return 0




def prepare_data(user_name, face_vectors, path="data/vector_data/"):
    """
    학습을 위한 {user_name} 사용자 얼굴 벡터 파일 로드 및 캡처한 얼굴 벡터 저장

    :param user_name: 사용자 이름
    :param face_vectors: 캡처된 사용자 얼굴 벡터
    :param path: 저장된 벡터 파일 경로
    :return: 학습 데이터, 검증 데이터, 테스트 데이터
    """
    # .npy 파일 경로 설정
    # 학습을 위한 {user_name} 사용자 얼굴 벡터 파일 로드 및 촬영한 얼굴 벡터 저장
    file_path = os.path.join(path, f'vector_data_{user_name}.npy')

    # 저장된 사용자 벡터 로드 (20, 128)
    user_vectors = np.load(file_path)
    # 입력 벡터를 (1, 128)로 변환
    input_vectors = face_vectors.reshape(1, -1)  # (1, 128)

    # 학습 데이터와 검증 데이터 분할
    X_train, X_val = train_test_split(user_vectors, test_size=0.2, random_state=42)

    # 데이터 확인용 메세지
    print(f'학습 데이터 크기: {X_train.shape}')            # (16, 128)
    print(f'검증 데이터 크기: {X_val.shape}')              # (4, 128)
    print(f'테스트 데이터 크기: {input_vectors.shape}')     # (1, 128)

    return X_train, X_val, input_vectors