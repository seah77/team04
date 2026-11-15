# 광고 문구 유형 분류기 (규칙 + 머신러닝)

건강기능식품·식품 광고를 OCR로 읽은 글에서 **문장마다 어떤 유형의 표현인지 분류**하고, 기능·효과 문장은 **식약처 공식 기능성 문구와 비교**합니다.

> 이 모듈은 위반 여부(과대광고 여부)를 **판정하지 않습니다.** 결과는 "질병 관련 표현일 수 있음", "공식 기능성 문구와 표현이 다름"처럼 **주의가 필요할 수 있는 표현을 안내**하는 용도입니다.

## 분류 유형 (5가지)

| 유형 | 예시 | 근거 |
|---|---|---|
| 질병 예방·치료 표현 | 치매 걱정 끝 / 당뇨 간식 / 변비에 좋은 차 | 식품표시광고법 제8조 1호 |
| 의약품 연상 표현 | 다이어트약 / 천연 수면유도제 / 마시는 위고비 | 같은 조 2호 |
| 기능·효과 표현 | 면역력 쑥쑥 / 붓기 쏙 / 키 컸어요 | 같은 조 3·4호 |
| 공식 기능성 문구 | 혈행 개선에 도움을 줄 수 있음 | 건강기능식품 인정 문구 형식 |
| 해당 없음 | 고소한 맛 / 1일 1포 / 포화지방 0g / 무료배송 | — |

규칙이 따로 붙이는 태그: 체험기·후기, 인증·보증 표방(식약처 인증, FDA 승인, 특허), 원료 효능 설명, 한약 처방명, "건강기능식품" 문구

## 처리 흐름

```
OCR 결과 (PaddleOCR / CLOVA OCR)
  → src/ocr_adapters.py : 좌표로 줄 정리, 신뢰도 낮은 조각 제거, 잘린 줄 이어 붙이기
  → 문장·해시태그 단위로 나누기
  → 문장마다
      ① 규칙 (src/ad_rules.py): 질병명, ~약/~제, 의약품명, 한약 처방명, 체험기, 인증, 영양성분표 항목
      ② 머신러닝: 글자 n-gram(TF-IDF) + 규칙 특징 → 로지스틱 회귀 → 5유형 + 확률
      ③ 규칙이 확실한 경우 규칙 우선 / 확률 0.45 미만이면 "확인 필요"
      ④ 기능·효과 문장 → 공식 기능성 문구와 비교: 확인됨 / 표현이 다름 / 확인되지 않음
  → 문장별 결과 (dict, JSON으로 바로 응답 가능)
```

## 폴더 구조

```
src/
  ad_rules.py               규칙 사전, 문장 나누기, 규칙 특징
  train_ad_classifier_v2.py 학습·평가·분석 (analyze_text)
  ocr_adapters.py           PaddleOCR / CLOVA OCR 결과 → 글 → 분석
  ocr_to_sheet.py           사진 폴더 → OCR → 라벨용 엑셀 (EasyOCR 또는 .txt)
models/ad_model_v2.joblib   학습된 모델 (scikit-learn 1.9.1)
data/labels/                라벨 엑셀 ('정답표' 시트) — 이 폴더의 .xlsx는 모두 학습에 사용
data/templates/             새 문장 수집용 빈 양식
data/official/              식약처 공식 데이터 (용량 때문에 저장소에 미포함, data/official/README.md 참고)
notebooks/train_colab.ipynb Colab 학습 노트북
examples/                   실제 OCR 결과 예시 (신뢰도<TAB>글자)
tests/check_examples.py     동작 확인 스크립트
reports/                    학습 결과 보고서, 오분류 목록, 기준값 표
```

## 빠른 시작

```bash
pip install -r requirements.txt
python tests/check_examples.py                     # 동작 확인 (모두 통과해야 함)
python src/ocr_adapters.py examples/clova_monster.tsv
python src/train_ad_classifier_v2.py --analyze "면역력 쑥쑥!
치매 걱정 끝
고소하고 담백한 맛"
```

`--analyze`, `tests`, 서버 사용에는 공식 데이터가 없어도 됩니다(모델 안에 공식 기능성 문구가 들어 있음). **다시 학습할 때만** `data/official/I0030.csv.gz`가 필요합니다.

## 서버(FastAPI)에 붙이기

```python
import sys; sys.path.insert(0, "ml/ad-claim-classifier/src")   # 저장소 위치에 맞게
from ocr_adapters import paddle_to_text, clova_to_text, analyze_image_text

text = paddle_to_text(paddle_result)          # PaddleOCR 원본 결과 (좌표 포함)
# text = clova_to_text(clova_response_json)   # CLOVA OCR 응답
results = analyze_image_text(text)            # 문장별 결과 리스트

# 건강기능식품으로 확인된 제품이면 그 제품의 인정 기능성(I0030 PRIMARY_FNCLTY)과 비교
results = analyze_image_text(text, product_fnclty=primary_fnclty)
```

결과 한 문장 예:
```json
{"text": "치매 걱정 끝", "type": "질병 예방·치료 표현", "ml_type": "질병 예방·치료 표현", "prob": 1.0,
 "rule_tags": [{"tag": "질병 예방·치료 표현", "evidence": "치매, 걱정 끝"}],
 "messages": ["질병의 예방·치료와 관련될 수 있는 표현입니다. 질병이 있다면 의사·약사와 상담하세요."]}
```
기능·효과 / 공식 기능성 문장에는 `"official": {"level": "표현이 다름", "nearest": "면역력 증진에 도움을 줄 수 있음", ...}`이 추가됩니다.

- 모델은 서버 시작 시 한 번만 불러옵니다(`get_bundle`이 캐시). GPU 불필요, 오프라인 동작.
- OCR 결과는 **좌표가 있는 원본**을 넘겨야 줄 순서가 맞습니다. 글자·신뢰도만 있으면 `scored_lines_to_text()`를 쓰되 순서가 섞일 수 있습니다.

## 다시 학습하기

```bash
python src/train_ad_classifier_v2.py
```
- `data/labels/`의 모든 엑셀 + 공식 기능성 문구 150개 + 일반식품 제품명 40개로 학습합니다.
- `학습/시험` 칸이 "시험"이고 사람이 라벨한 문장이 20개 이상이면 그것으로 평가하고, 아니면 학습 데이터의 20%를 임시 시험셋으로 씁니다.
- 결과는 `reports/`에, 모델은 `models/`에 저장됩니다.
- Colab: `notebooks/train_colab.ipynb`

라벨 엑셀 규칙: `최종 라벨` → (`라벨러1` = `라벨러2`) → 한 명만 있으면 그 값 → 둘 다 비면 `초안 라벨`. 라벨러1·2가 다르고 최종 라벨이 없으면 학습에서 빠집니다.

## 데이터

| 파일 | 내용 | 문장 수 |
|---|---|---|
| `data/labels/casebook_phrases.xlsx` | 식약처 온라인 부당광고 사례집(2022, 2024~25)에 인용된 위반 문구 | 328 |
| `data/labels/casebook_captures.xlsx` | 사례집(2024~25) 광고 캡처 이미지 속 문장을 옮긴 것 (가격·배송 등 주변 문장 포함) | 370 |
| `data/labels/synthetic_none.xlsx` | "해당 없음" 보강 문장 (직접 작성한 합성 문장, 학습 전용) | 110 |
| 공식 데이터 (자동 추가) | I0030 공식 기능성 문구 150 / I1250 일반식품 제품명 40 | 190 |

라벨은 사례집의 법 조항 분류를 기준으로 **AI(Claude)가 붙였습니다.** 사람 검수 전이며, 특히 시험셋은 팀원 검수가 필요합니다.

## 성능 (현재)

| 평가 | 결과 |
|---|---|
| 교차검증 (학습 데이터 안, macro F1) | 글자만 0.88 → 글자+규칙 0.92~0.93 |
| **실제 광고 문장 97개 (사례집 캡처, 규칙 보강 전)** | **정답률 82%, macro F1 0.79** |
| 같은 97개, 규칙 보강 후 | macro F1 0.97 (틀린 문장을 보고 고친 뒤라 **참고용**) |
| 휴대폰 사진 OCR | **아직 미평가** |

알려진 한계
- 사례집 문장은 이미 텍스트라 OCR 오류가 없음 → 휴대폰 사진에서는 더 낮을 수 있음
- 기능을 돌려 말하는 문장("집중력이 필요한 모든 순간에", "체온이 올라가요")은 놓치기 쉬움
- 한약 처방명 사전은 영문 고시 별표1을 한글로 옮긴 것 → 한국어 고시와 대조 필요
- 라벨은 AI 1인 기준

## 다음 할 일
1. 건강식품·건기식 앞면(효능 문구가 있는 면) 사진 10~15장 → OCR → `src/ocr_to_sheet.py`로 시험셋 작성 → 사람 라벨
2. 시험셋 라벨 사람 검수
3. 틀린 문장(`reports/misclassified.csv`) 보고 데이터·규칙 보강

## 출처
- 식품의약품안전처, 「식품 분야 온라인 부당광고 사례집」(2024~25), 「온라인 식품·건강기능식품·의약외품·화장품 분야 부당광고 사례집」(2022)
- 식품의약품안전처, 「식품등의 부당한 표시 또는 광고의 내용 기준」 고시
- 식품안전나라 Open API: I0030(건강기능식품 품목제조신고), I-0040(개별인정형 원료), I1250(식품 품목제조보고)
