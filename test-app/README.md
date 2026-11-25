# 광고 문구 분석 테스트 앱 (CLOVA OCR + 머신러닝)

제품 사진(앞면·뒷면)을 찍으면 → CLOVA OCR로 글자를 읽고 → 광고 문구 유형을 분류해서 → 식약처 공식 정보와 같은지/다른지 보여 주는 **시험용 앱**입니다.
판정(위반·안전·효과)은 하지 않고, 공식 정보와의 비교만 보여 줍니다.

```
[휴대폰 앱 (Expo)]  사진 1~4장 → POST /analyze_multi_base64
        ▼
[분석 서버 (FastAPI)]  server/ocr_server.py
   ① CLOVA OCR (사진 여러 장 동시에)
   ② 제품 확인: 품목보고번호 → 제품명 후보 → 확인 불가   (data/official)
   ③ 문장 분류: 규칙 + 머신러닝                          (../ad-claim-classifier)
   ④ 공식 기능성 문구·원료 인정 기능성과 비교
   ⑤ 안내 문구                                            (../ad-claim-classifier/data/messages.json)
        ▼
[휴대폰 앱]  한 줄 요약 + 제품 카드 + 원료 카드 + 문장별 결과
```

## 폴더

```
저장소/
  ad-claim-classifier/     머신러닝 (feat/ml-classifier) — 서버가 자동으로 찾아 씀
  test-app/
    server/
      ocr_server.py        분석 서버
      clova_test.py        OCR만 따로 시험 (사진 폴더 → 글자)
      requirements.txt
      .env.example         키 넣는 양식 (.env 는 올리지 않음)
    app_expo/              React Native + Expo 앱 (App.js, package.json, app.json, assets/)
    data/official/         공식 데이터 넣는 곳 (깃허브에는 없음, README 참고)
```

## 실행 (Windows, cmd)

### 1. 서버 준비 (처음 한 번)
```
cd test-app\server
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```
`.env`를 메모장으로 열어 `CLOVA_OCR_URL`, `CLOVA_OCR_SECRET`을 채웁니다. **`.env`와 키는 절대 깃허브에 올리지 않습니다.**
공식 데이터 두 파일을 `test-app\data\official\`에 넣습니다.

### 2. 서버 켜기
```
cd test-app\server
.venv\Scripts\activate.bat
uvicorn ocr_server:app --host 0.0.0.0 --port 8000
```
확인: 브라우저에서 `http://localhost:8000/health` → `{"ok":true, ...}`

### 3. 앱 켜기 (처음이면 `npm install` 먼저)
```
cd test-app\app_expo
npm install
npx expo start -c
```
- 안드로이드: Expo Go 앱에서 QR 스캔
- 아이폰: 기본 카메라로 QR 스캔 → Expo Go에서 열기 → "로컬 네트워크" 허용
- 휴대폰과 노트북은 **같은 와이파이**여야 합니다. 앱의 서버 주소 칸에 `http://노트북IP:8000` (`ipconfig`의 IPv4 주소)

## 서버 API

| 주소 | 설명 |
|---|---|
| `GET /health` | 서버 상태 |
| `POST /analyze_multi_base64` | 사진 여러 장 `{"images":[{"image":"<base64>","format":"jpg"}, ...]}` (최대 4장) |
| `POST /analyze_base64` | 사진 한 장 `{"image":"<base64>","format":"jpg"}` |
| `POST /analyze_text` | OCR 없이 글로 시험 `{"text":"..."}` 또는 `{"texts":["앞면","뒷면"]}` (CLOVA 요금 없음) |
| `GET /reload_messages` | 안내 문구 파일(messages.json)을 고친 뒤 서버를 끄지 않고 다시 읽기 |

## 안내 문구 바꾸기
`ad-claim-classifier/data/messages.json`의 오른쪽 글자만 고칩니다. `{official}` 같은 중괄호는 지우지 않습니다.
원칙: 판정하지 않기(불법·위반·허위·과대광고·효과 없음·안전함 금지), 공식 정보와 같다·다르다·없다만 말하기.

## 알려진 한계
- 제품명 검색은 건강기능식품만 대상이고 후보일 뿐이라 사용자 확인이 필요합니다. 뒷면(품목보고번호)을 같이 찍으면 정확합니다.
- 원료 인식은 고시형 32개 + 개별인정형 48개만 알아봅니다.
- 화면을 다시 찍은 사진, 흐린 사진은 OCR 잡음이 많습니다.
- 서버가 노트북에 있어서 같은 와이파이에서만 됩니다 (배포 전 시험용).
