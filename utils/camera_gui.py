import os
import cv2 as cv
from tkinter import *           # Tkinter 라이브러리: GUI 생성
from PIL import Image, ImageTk  # PIL(pillow) : OpenCV 이미지를 Tkinter에서 표시하기 위한 변환


class CameraApp:
    def __init__(self, root):
        """
        CameraApp 클래스 생성자
        GUI 초기화 수행 및 OpenCV를 통해 카메라 연결
        """

        self.root = root                    # root : Tkinter 창으로, 여기에 GUI를 추가
        self.root.title("Camera GUI App")   # 창 제목 설정

        # 0 : 연결된 첫 번째 웹캠 ID
        # 카메라를 초기화하고 영상 데이터를 가져올 준비
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            exit()

        # Tkinter의 Label : 카메라 영상을 표시할 영역 할당
        self.video_frame = Label(self.root)
        # grid 레이아웃 : GUI 구성 요소 배치
        ## Label 위젯을 (0, 0) 위치에 배치
        ## columnspan=2 : Label이 두 열을 차지
        self.video_frame.grid(row=0, column=0, columnspan=2)

        # 촬영 버튼
        ## 클릭 시 capture_image 메서드 실행
        self.capture_button = Button(self.root, text="촬영", command=self.capture_image)
        self.capture_button.grid(row=1, column=0, pady=10)
        # 종료 버튼
        ## 클릭시 close_app 메서드 실행
        self.close_button = Button(self.root, text="종료", command=self.close_app)
        self.close_button.grid(row=1, column=1, pady=10)

        # 캡처 이미지 저장 변수 (초기값 None)
        ## 사용자가 캡처한 이미지 저장
        self.captured_image = None
        ## 영상 스트리밍 상태 관리(False: 캡처 전이므로 계속 스트리밍)
        self.is_captured = False

        # 실사간으로 카메라 영상 업데이트 메서드
        self.update_video_frame()


    def capture_image(self):
        """
        촬영 버튼 클릭 시 현재 프레임을 캡처하고, 화면에 고정
        이후 고유 파일 이름으로 저장

        OpenCV는 BGR 형식으로, Tkinter와 Pillow는 RGB 형식으로 이미지를 처리하기 때문에
        이미지 형태를 BGR -> RGV로 변환
        """
        # 객체 self.cap을 이용해 현재 카메라 프레임 읽기
        ## 캡처된 프레임은 Numpy 배열 형태로 반환
        ## read() 메서드 반환 값 2가지
        ### ret : 프레임이 성공적으로 읽혔는지 여부를 나타내는 Boolean 값
        ### frame : 카메라에서 읽어 온 현재 프레임 (BGR 형태)
        ret, frame = self.cap.read()

        if ret:
            # 캡처된 프레임 저장 (BGR 형식)
            self.captured_image = frame
            # 캡처 후 상태로 전환하며 스트리밍 중단
            self.is_captured = True
            print('이미지 촬영이 완료되었습니다.')

            # 캡처한 이미지를 Tkinter Label에 표시 (GUI 화면에 고정)
            ## OpenCV의 BGR 이미지(Numpy 배열 이미지 형식)를 RGB 형식으로 변환
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            ## Tkinter는 OpenCV의 Numpy 배열 이미지를 지원하지 않으므로 이를 위한 변환
            ### Image.fromarray(frame) : Numpy 배열 이미지를 Pillow의 Image 객체로 변환
            ### ImageTK.PhotoImage : Pillow의 Image 객체를 Tkinter에서 사용 가능한 객체로 변환
            #### 이를 통해 Tkinter의 Label 위젯에 이미지를 표시할 수 있게 됨
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Tkinter Label에 형식이 변화된 이미지를 표시
            self.video_frame.config(image=self.photo)
        else:
            print('프레임을 읽지 못했습니다.')
            return


    def close_app(self):
        """
        종료 버튼 클릭 또는 얼굴 인식 완료 시 Tkinter 창을 닫고 리소스 해제
        """
        self.cap.release()      # 카메라 리소스 해제
        self.root.quit()        # Tkinter 루프 종료
        self.root.destroy       # Tkinter 창 닫기


    def get_captured_image(self):
        """
        캡처된 이미지를 반환

        :return: OpenCV BGR 이미지 또는 None
        """
        return self.captured_image


    def update_video_frame(self):
        """
        카메라에서 실시간 프레임을 읽어와 Tkinter Label에 출력
        """
        # 캡처 상태가 아니라면 계속 업데이트
        if not self.is_captured:
            ret, frame = self.cap.read()
            # 프레임 단위로 영상이 잘 읽어지고 있다면
            if ret:
                # OpenCV의 BGR 이미지(Numpy 배열 이미지 형식)를 RGB 형식으로 변환
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                ## Tkinter는 OpenCV의 Numpy 배열 이미지를 지원하지 않으므로 이를 위한 변환
                ### Image.fromarray(frame) : Numpy 배열 이미지를 Pillow의 Image 객체로 변환
                ### ImageTK.PhotoImage : Pillow의 Image 객체를 Tkinter에서 사용 가능한 객체로 변환
                #### 이를 통해 Tkinter의 Label 위젯에 이미지를 표시할 수 있게 됨
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                # Tkinter Label에 형식이 변화된 이미지를 표시
                self.video_frame.config(image=self.photo)
        # 10ms 후 다시 호출해 실시간 영상 업데이트 유지
        self.root.after(10, self.update_video_frame)


def run_CameraApp():
    """
    CameraApp GUI를 실행하고, 캡처된 이미지를 반환

    :return: OpenCV BGR 이미지 또는 None
    """
    # Tkinter root 창 생성
    root = Tk()
    # root를 매개변수로 전달하여 CameraApp 클래스 인스턴스 생성
    app = CameraApp(root)
    # Tkinter 루프 실행 (GUI 띄우기)
    root.mainloop()

    # GUI 종료 후 캡처 이미지 반환
    return app.get_captured_image()


def get_img_name(base_name="captured_img", extension=".jpg", path="../data/captured_images/"):
    """
    고유한 파일 이름을 생성하는 함수

    :param base_name: 기본 파일 이름 (str)
    :param extension: 파일 확장자 (str)
    :param folder: 저장 경로 (str)
    :return: 고유한 파일 이름 (str)
    """
    i = 1
    while True:
        # 파일 이름 생성
        filename = os.path.join(f'{path}{base_name}{i:02d}{extension}')

        # 설정된 저장 경로에 동일한 이름의 파일이 존재하지 않으면 저장
        if not os.path.exists(filename):
            return filename
        i += 1

