from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np

# 서비스 파일들 임포트
from ..services.image_service import preprocess_image
from ..services.ocr_service import call_external_ocr
# 👇 아래처럼 분석 함수를 임포트해 주어야 합니다! (어느 파일에 정의했느냐에 따라 경로를 확인해 주세요)
from ..services.analysis_service import postprocess_and_analyze_with_llm  

router = APIRouter(prefix="/api/v1", tags=["OCR & Validation Pipeline"])

@router.post("/ocr-scan")
async def scan_food_label(file: UploadFile = File(...)):
    """
    1. 이미지 업로드 유효성 검사
    2. OpenCV 이미지 전처리 (image_service)
    3. 외부 OCR API 텍스트 추출 (ocr_service)
    4. LLM 오탈자 교정 및 성분/광고 분석 (analysis_service)
    """
    # 1. 이미지 파일 형식 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 파일 바이트 읽기
        image_bytes = await file.read()
        
        # 2. 이미지 전처리 수행
        processed_image = preprocess_image(image_bytes)
        
        # 3. OCR 텍스트 추출 수행 (현재는 가상 텍스트 반환)
        raw_text = call_external_ocr(processed_image)
        
        # 4. LLM 후처리 및 분석 수행 
        # (주의: 실제 OpenAI API 키를 넣기 전에는 에러가 날 수 있으므로, 
        #  키 설정 전이라면 아래 analysis_result를 임시 딕셔너리로 대체해도 좋습니다)
        try:
            analysis_result = postprocess_and_analyze_with_llm(raw_text)
        except Exception as e:
            # API 키가 아직 설정되지 않은 경우를 대비한 임시 예외 처리
            analysis_result = f"LLM 분석 보류 (API 키 필요): {str(e)}"

        # 5. 최종 결과 반환
        return {
            "status": "success",
            "filename": file.filename,
            "raw_text": raw_text,
            "analysis": analysis_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")