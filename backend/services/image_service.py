import cv2
import numpy as np

def preprocess_image(image_bytes: bytes):
    """
    사용자가 업로드한 이미지 바이트 데이터를 받아 
    OpenCV를 사용해 흑백 변환 및 이진화(빛반사/그림자 보정)를 진행합니다.
    """
    # 1. 바이트 데이터를 이미지 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. 흑백(Grayscale)으로 변환 (글씨와 배경 대비 극대화)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 이진화 (Thresholding) - 빛반사나 그림자 영역을 선명하게 조절
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 전처리된 이미지(배열) 반환
    return thresh