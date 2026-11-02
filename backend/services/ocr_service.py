import cv2
import os
from google.cloud import vision

# 프로젝트 루트 경로를 기준으로 credentials.json 파일을 안전하게 찾습니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

def call_external_ocr(processed_image) -> str:
    """
    OpenCV로 전처리된 이미지(numpy 배열)를 받아 
    Google Cloud Vision API로 전송하고, 추출된 텍스트를 반환합니다.
    """
    # 1. OpenCV 이미지 배열을 JPEG 바이트로 인코딩
    success, encoded_image = cv2.imencode('.jpg', processed_image)
    if not success:
        raise ValueError("이미지 인코딩에 실패했습니다.")
    
    image_bytes = encoded_image.tobytes()

    # 2. Google Cloud Vision 클라이언트 초기화
    client = vision.ImageAnnotatorClient()

    # 3. Vision API에 보낼 이미지 객체 생성
    image = vision.Image(content=image_bytes)

    # 4. 텍스트 감지(TEXT_DETECTION) 요청 수행
    response = client.text_detection(image=image)
    
    if response.error.message:
        raise Exception(f"Vision API 오류 발생: {response.error.message}")

    texts = response.text_annotations
    
    if not texts:
        return "감지된 텍스트가 없습니다."

    extracted_text = texts[0].description
    
    return extracted_text