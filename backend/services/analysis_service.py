from openai import OpenAI  # 또는 구글 GenAI SDK 등 사용하는 LLM 라이브러리

def postprocess_and_analyze_with_llm(raw_ocr_text: str):
    """
    OCR로 추출된 날것의 텍스트를 받아 LLM을 통해 오탈자를 교정하고,
    알레르기 유발 성분이나 주의사항을 분석합니다.
    """
    # 본인의 API 키 또는 환경 변수(os.getenv)에서 키를 불러옵니다.
    client = OpenAI(api_key="OPENAI_API_KEY")  # 환경 변수에서 불러오는 것이 안전합니다.

    prompt = f"""
    아래 텍스트는 식품 라벨을 OCR로 추출한 날것의 데이터야. 오탈자가 있을 수 있으니 문맥상 올바르게 교정하고, 
    알레르기 유발 성분이 있는지 분석해줘.
    
    [OCR 텍스트]
    {raw_ocr_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",  # 또는 프로젝트에서 사용하는 모델 명
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content