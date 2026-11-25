# -*- coding: utf-8 -*-
"""
테스트용 분석 서버: 앱 → (이 서버) → CLOVA OCR → 광고 문구 분석 → 앱

왜 서버를 거치나
  CLOVA Secret Key를 앱 안에 넣으면 앱 파일에서 키를 꺼낼 수 있습니다.
  키는 이 서버의 환경변수에만 두고, 앱은 이 서버에 사진만 보냅니다.

준비
  pip install fastapi uvicorn python-multipart python-dotenv requests pandas scikit-learn==1.9.1 scipy openpyxl joblib
  폴더 배치:
    ml/ad-claim-classifier/   (깃허브의 분류기 폴더)
    server/ocr_server.py      (이 파일)
    data/official/I0030.csv.gz, I1250_csv.gz   (선택: 품목보고번호 조회용)

환경변수 (네이버 클라우드 콘솔 → CLOVA OCR → 도메인 → API Gateway 연동에서 확인)
  Windows PowerShell:
    $env:CLOVA_OCR_URL="https://....apigw.ntruss.com/custom/v1/..../general"
    $env:CLOVA_OCR_SECRET="발급받은 Secret Key"
  Mac/Linux:
    export CLOVA_OCR_URL=...   export CLOVA_OCR_SECRET=...

실행 (같은 와이파이의 휴대폰에서 접속하려면 --host 0.0.0.0)
  uvicorn ocr_server:app --host 0.0.0.0 --port 8000
  확인: 브라우저에서 http://<노트북 IP>:8000/docs  → /analyze 에 사진 올려 보기
"""
import json, os, re, sys, time, uuid
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 머신러닝 코드 위치: ① 환경변수 ML_SRC ② 테스트앱 묶음(ad_test_app/ml/…) ③ 깃허브 저장소(저장소/ad-claim-classifier/…)
_ML_CANDIDATES = [os.environ.get('ML_SRC', ''),
                  os.path.join(ROOT, 'ml', 'ad-claim-classifier', 'src'),
                  os.path.join(os.path.dirname(ROOT), 'ad-claim-classifier', 'src')]
ML_SRC = next((p for p in _ML_CANDIDATES if p and os.path.isfile(os.path.join(p, 'ocr_adapters.py'))), _ML_CANDIDATES[1])
DATA = os.environ.get('OFFICIAL_DIR', os.path.join(ROOT, 'data', 'official'))
sys.path.insert(0, ML_SRC)
from ocr_adapters import clova_to_text, analyze_image_text, get_bundle   # noqa: E402
from ingredients import find_ingredients, public as public_ingredients  # noqa: E402
from messages import msg, reload as reload_messages                                                  # noqa: E402  (문구: ml/.../data/messages.json)
import ad_rules as R                                                      # noqa: E402

try:
    from dotenv import load_dotenv          # server/.env 파일이 있으면 읽음
    load_dotenv(os.path.join(HERE, '.env'))
except ImportError:
    pass
CLOVA_URL = os.environ.get('CLOVA_OCR_URL', '')
CLOVA_SECRET = os.environ.get('CLOVA_OCR_SECRET', '')


# ------------------------------------------------------------------ CLOVA OCR 호출
def call_clova(image_bytes, fmt='jpg', timeout=30):
    """사진 바이트 → CLOVA General OCR 응답(dict). base64 JSON 방식(조원이 시험해서 동작 확인한 방식)."""
    if not CLOVA_URL or not CLOVA_SECRET:
        raise RuntimeError('CLOVA_OCR_URL / CLOVA_OCR_SECRET 환경변수(.env)를 설정하세요.')
    import base64
    body = {'version': 'V2', 'requestId': str(uuid.uuid4()), 'timestamp': int(time.time() * 1000),
            'lang': 'ko', 'images': [{'format': fmt, 'name': 'upload', 'data': base64.b64encode(image_bytes).decode()}]}
    r = requests.post(CLOVA_URL, json=body, headers={'X-OCR-SECRET': CLOVA_SECRET}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f'CLOVA OCR 오류 {r.status_code}: {r.text[:300]}')
    return r.json()


# ------------------------------------------------------------------ 제품 확인: 품목보고번호 → 제품명 후보
_HF = _GF = _NAME_IDX = None
def GENERAL_FOOD_NOTICE():
    return msg('product.general_food')


def _load_official():
    global _HF, _GF
    if _HF is not None:
        return
    import pandas as pd
    p1, p2 = os.path.join(DATA, 'I0030.csv.gz'), os.path.join(DATA, 'I1250_csv.gz')
    _HF = (pd.read_csv(p1, dtype=str, usecols=['PRDLST_REPORT_NO', 'PRDLST_NM', 'BSSH_NM', 'PRIMARY_FNCLTY',
                                                'IFTKN_ATNT_MATR_CN']).drop_duplicates('PRDLST_REPORT_NO')
           .set_index('PRDLST_REPORT_NO') if os.path.exists(p1) else {})
    _GF = (pd.read_csv(p2, dtype=str, usecols=['PRDLST_REPORT_NO', 'PRDLST_NM', 'BSSH_NM', 'PRDLST_DCNM'])
           .drop_duplicates('PRDLST_REPORT_NO').set_index('PRDLST_REPORT_NO') if os.path.exists(p2) else {})
    _build_name_index()


REPORT_NO = re.compile(r'품목\s*(?:보고|제조)?\s*번호\s*[:：]?\s*([0-9][0-9\s\-]{9,20})')


def find_report_no(text):
    m = REPORT_NO.search(text)
    if m:
        return re.sub(r'\D', '', m.group(1))
    m = re.search(r'(?<!\d)(\d{12,16})(?!\d)', re.sub(r'(?<=\d)[\s\-](?=\d)', '', text))   # 번호만 따로 찍힌 경우
    return m.group(1) if m else None


def _hf_record(no, how, score=None):
    r = _HF.loc[no]
    d = {'분류': '건강기능식품', '확인 방법': how, '품목보고번호': no, '제품명': r['PRDLST_NM'], '업체': r['BSSH_NM'],
         '기능성': r['PRIMARY_FNCLTY'], '섭취시주의사항': r['IFTKN_ATNT_MATR_CN']}
    if score is not None:
        d['일치도'] = score
    return d


def lookup(no):
    """품목보고번호 → 건강기능식품(I0030) / 일반식품(I1250) / 확인 불가"""
    _load_official()
    if len(_HF) and no in _HF.index:
        d = _hf_record(no, '품목보고번호')
        d['안내'] = msg('product.hf_by_report_no')
        return d
    if len(_GF) and no in _GF.index:
        r = _GF.loc[no]
        return {'분류': '일반식품', '확인 방법': '품목보고번호', '품목보고번호': no, '제품명': r['PRDLST_NM'],
                '업체': r['BSSH_NM'], '식품유형': r['PRDLST_DCNM'], '안내': GENERAL_FOOD_NOTICE()}
    return {'분류': '확인 불가', '품목보고번호': no,
            '안내': msg('product.report_no_not_found')}


# ---- 제품명 후보 검색 (건강기능식품 I0030 기준)
# 흔한 말·효능 단어는 가중치를 낮춰, '혈당관리' 같은 광고 문구만으로 엉뚱한 제품이 잡히지 않게 함
GENERIC = set("""유산균 프로바이오틱스 비타민 홍삼 건강 케어 플러스 골드 프리미엄 데일리 맥스 파워 프로 슈퍼 더블 트리플
              캡슐 정 분말 스틱 앤 and the plus gold 키즈 맘 우먼 맨 시니어 액 젤리 츄어블 멀티 종합 오메가3 루테인
              혈당 관리 면역 다이어트 콜레스테롤 혈행 관절 수면 피로 간 장 눈 뼈 키 뇌 코 위 피부
              간편한 건강한 맛있는 오늘 하루 매일 순수 자연 천연 국내산 유기농 오직 진짜 착한 좋은 바른 든든
              온가족 가족 어린이 아이 엄마 아빠 부모님 선물 세트 기획 특가 신제품 리뉴얼 국산 한국 코리아""".split())


def _is_generic(t):
    """흔한 말 또는 효능 단어('혈당관리', '면역케어')면 True → 이 단어만으로는 제품을 고르지 않음"""
    from ad_rules import targets
    from train_ad_classifier_v2 import FUNC_WORD
    return t in GENERIC or bool(targets(t)) or bool(FUNC_WORD.search(t))


def _tokens(s):
    return [t for t in re.split(r'[\s()\[\],/·&+_:;\'"‘’“”]+', str(s).lower()) if t]


def _nospace(s):
    return re.sub(r'[^0-9a-z가-힣]', '', str(s).lower())


def _build_name_index():
    global _NAME_IDX
    _NAME_IDX = {'tok': {}, 'items': []}
    if not len(_HF):
        return
    seen = set()
    for no, name in zip(_HF.index, _HF['PRDLST_NM']):
        name = str(name)
        key = _nospace(name)
        if not key or key in seen:          # 같은 이름은 하나만 (가장 앞의 품목보고번호)
            continue
        seen.add(key)
        toks = [t for t in _tokens(name) if len(t) >= 2 or re.match(r'[가-힣]', t)]
        k = len(_NAME_IDX['items'])
        _NAME_IDX['items'].append((no, name, toks, key))
        gen = {t: _is_generic(t) for t in set(toks)}
        _NAME_IDX['items'][-1] = (no, name, toks, key, gen)
        for t in set(toks):
            if not gen[t] and len(t) >= 2:
                _NAME_IDX['tok'].setdefault(t, []).append(k)


def search_by_name(text, top=3, min_score=0.62):
    """OCR 글 → 건강기능식품 제품명 후보 [(일치도, 품목보고번호, 제품명, 순위점수)]"""
    _load_official()
    if not _NAME_IDX or not _NAME_IDX['items']:
        return []
    ocr_toks = set(_tokens(text)); ocr_ns = _nospace(text)
    cand = set()
    for t in ocr_toks:
        cand.update(_NAME_IDX['tok'].get(t, []))
    scored = []
    for k in cand:
        no, name, toks, key, gen = _NAME_IDX['items'][k]
        w = [0.5 if gen[t] else 1.0 for t in toks]
        hit = sum(wi for t, wi in zip(toks, w) if t in ocr_toks or (len(t) >= 2 and t in ocr_ns))
        score = hit / sum(w) if w else 0
        contiguous = len(key) >= 4 and key in ocr_ns          # 제품명 전체가 붙어서 그대로 나옴
        if sum(w) < 1.5 and not (contiguous and len(key) >= 5):
            continue                                           # '간편한'처럼 한 단어짜리 이름은 우연히 맞기 쉬움
        # 제품을 특정할 만한 단어(브랜드·고유 이름: 한글 3자 이상 또는 영문·숫자 4자 이상)가 하나는 맞아야 함
        anchor = any(not gen[t] and (t in ocr_toks or t in ocr_ns) and
                     (len(t) >= 3 if re.search('[가-힣]', t) else len(t) >= 4) for t in toks)
        if not anchor and not (contiguous and len(key) >= 6):
            continue
        rank = score + (0.2 if contiguous else 0)
        if score >= min_score:
            scored.append((rank, hit, len(name), round(min(1.0, rank), 2), no, name))
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    return [(sc, no, name, rk) for rk, _, _, sc, no, name in scored[:top]]


def identify_product(text):
    """품목보고번호가 있으면 번호로, 없거나 못 찾으면 제품명 후보로 확인"""
    _load_official()
    no = find_report_no(text)
    if no:
        r = lookup(no)
        if r['분류'] != '확인 불가':
            return r
    cands = search_by_name(text)
    if cands:
        top = cands[0]
        others = [{'제품명': n, '품목보고번호': x, '일치도': s} for s, x, n, _ in cands[1:]]
        clear = len(cands) == 1 or top[3] - cands[1][3] >= 0.1
        r = _hf_record(top[1], '제품명 후보', top[0])
        r['다른 후보'] = others
        r['안내'] = msg('product.hf_by_name_clear' if clear else 'product.hf_by_name_ambiguous')
        return r
    if no:
        return lookup(no)
    return {'분류': '확인 불가', '안내': msg('product.not_found')}


# ------------------------------------------------------------------ 전체 처리
def analyze_bytes(image_bytes, fmt='jpg', clova_response=None):
    t0 = time.time()
    resp = clova_response or call_clova(image_bytes, fmt)
    text = clova_to_text(resp)
    return analyze_ocr_text(text, t0)


def headline(sentences, kind):
    """결과 맨 위 한 줄 요약 (판정 없이 개수만)"""
    ads = [s for s in sentences if not s.get('is_product_name') and not s.get('is_label_info')]
    dd = sum(s['type'] in ('질병 예방·치료 표현', '의약품 연상 표현') for s in ads)
    miss = sum(kind == '건강기능식품' and bool((s.get('official') or {}).get('targets'))
               and s['official'].get('level') in ('확인되지 않음', '일부만 확인') for s in ads)
    unclear = sum(s['type'] == '확인 필요' for s in ads)
    fx = sum(s['type'] in ('기능·효과 표현', '공식 기능성 문구') for s in ads)
    parts = [msg('summary.disease_drug', k=dd) if dd else '',
             msg('summary.function', k=fx) if fx else '',
             msg('summary.not_in_product', k=miss) if miss else '',
             msg('summary.unclear', k=unclear) if unclear else '']
    parts = [p for p in parts if p]
    return msg('summary.counted', n=len(ads)) + ' ' + (msg('summary.found', parts=', '.join(parts)) if parts else msg('summary.nothing'))


def mark_product_name(sentences, name):
    """'바이크롬 간 유산균'처럼 제품명 그대로인 줄은 광고 문구가 아니므로 분석에서 뺌"""
    if not name:
        return
    nn = re.sub(r'[\s()\[\]·,]', '', name)
    for s in sentences:
        t = re.sub(r'[\s()\[\]·,]', '', s.get('text', ''))
        if len(t) >= 3 and (t == nn or (t in nn and len(t) >= len(nn) * 0.6) or (nn in t and len(t) <= len(nn) + 3)):
            s.update({'type': '해당 없음', 'is_product_name': True, 'official': None,
                      'messages': [msg('sentence.product_name')], 'ingredient_refs': []})


def ingredient_summary(text, product_fnclty=None):
    """광고 글에 나온 원료 → 원료별 일반 인정 기능성 + 광고에서 말한 대상(눈·간·장 …)과 비교"""
    ad_targets = set(R.targets(text))
    prod_targets = set(R.targets(product_fnclty)) if product_fnclty else None
    out = []
    for g in public_ingredients(find_ingredients(text)):
        g = dict(g)
        g['ad_mentions'] = sorted(ad_targets & set(g.get('targets', [])))   # 광고에서 말한 것 중 이 원료 기능성에 있는 것
        if prod_targets is not None:                                          # 제품이 확인된 경우: 이 제품 인정 기능성에도 있는지
            g['in_product'] = sorted(set(g.get('targets', [])) & prod_targets)
        out.append(g)
    return out


def analyze_ocr_text(text, t0=None):
    """글 하나(사진 한 장) 분석"""
    return analyze_ocr_texts([text], t0)


def analyze_ocr_texts(texts, t0=None):
    """사진 여러 장(앞면·뒷면 …)의 OCR 글을 한 제품으로 묶어 분석.
    제품 확인·원료 인식은 모든 사진의 글을 합쳐서 하고, 문장은 사진별로 나눠 분석함."""
    t0 = t0 or time.time()
    texts = [t for t in texts if t and t.strip()] or ['']
    full = '\n'.join(texts)
    product = identify_product(full)                     # 품목보고번호는 보통 뒷면에 있음 → 합친 글로 찾기
    kind = product['분류'] if product['분류'] in ('건강기능식품', '일반식품') else None
    fn = product.get('기능성') if kind == '건강기능식품' else None
    sentences = []
    for n, t in enumerate(texts, 1):
        for s in analyze_image_text(t, product_fnclty=fn, product_kind=kind, context_text=full):
            if len(texts) > 1:
                s['photo'] = n                            # 몇 번째 사진에서 나온 문장인지
            sentences.append(s)
    mark_product_name(sentences, product.get('제품명'))
    ocr_text = texts[0] if len(texts) == 1 else '\n\n'.join(f'[사진 {n}]\n{t}' for n, t in enumerate(texts, 1))
    return {'ocr_text': ocr_text, 'photo_count': len(texts), 'product': product, 'sentences': sentences,
            'ingredients': ingredient_summary(full, fn),
            'summary': {'문장 수': len(sentences),
                        '주의 표현 수': sum(s['type'] in ('질병 예방·치료 표현', '의약품 연상 표현') for s in sentences),
                        '기능·효과 표현 수': sum(s['type'] == '기능·효과 표현' for s in sentences),
                        '확인 필요': sum(s['type'] == '확인 필요' for s in sentences)},
            'elapsed_sec': round(time.time() - t0, 2),
            'headline': headline(sentences, kind),
            'labels': {'ingredient_card_title': msg('ingredient.card_title'),
                       'ingredient_card_note': msg('ingredient.card_note'),
                       'caution_title': msg('product.caution_title')},
            'notice': msg('notice')}


def analyze_many_bytes(images):
    """[(사진 바이트, 'jpg'), ...] → CLOVA OCR을 동시에 돌린 뒤 한 제품으로 분석"""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(4, len(images))) as ex:
        texts = list(ex.map(lambda x: clova_to_text(call_clova(x[0], x[1])), images))
    return analyze_ocr_texts(texts, t0)


# ------------------------------------------------------------------ FastAPI
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title='광고 문구 분석 테스트 서버')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

    @app.on_event('startup')
    def _warmup():
        get_bundle()          # 모델 미리 불러오기
        _load_official()      # 공식 데이터 미리 불러오기 (없으면 건너뜀)

    @app.get('/reload_messages')
    def reload_msgs():
        """messages.json 을 고친 뒤 서버를 끄지 않고 다시 읽기: 브라우저에서 http://<서버>:8000/reload_messages"""
        reload_messages()
        return {'ok': True}

    @app.get('/health')
    def health():
        return {'ok': True, 'clova_configured': bool(CLOVA_URL and CLOVA_SECRET)}

    @app.post('/analyze')
    async def analyze(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise HTTPException(400, '빈 파일입니다.')
        ext = (os.path.splitext(file.filename or '')[1].lstrip('.') or 'jpg').lower()
        fmt = 'jpg' if ext in ('jpeg', 'jpg', 'heic') else ext
        try:
            return analyze_bytes(data, fmt)
        except RuntimeError as e:
            raise HTTPException(502, str(e))

    @app.post('/analyze_base64')
    def analyze_base64(payload: dict):
        """앱에서 사진을 base64 글자로 보내는 방식: {"image": "<base64>", "format": "jpg"}
        (최신 Expo의 fetch는 FormData 파일 업로드를 지원하지 않아서 이 방식을 씀)"""
        import base64, binascii
        b64 = payload.get('image', '')
        if ',' in b64[:100]:                       # 'data:image/jpeg;base64,...' 형태면 앞부분 제거
            b64 = b64.split(',', 1)[1]
        try:
            data = base64.b64decode(b64)
        except (binascii.Error, ValueError):
            raise HTTPException(400, 'base64 이미지가 올바르지 않습니다.')
        if not data:
            raise HTTPException(400, '빈 이미지입니다.')
        fmt = str(payload.get('format', 'jpg')).lower().replace('jpeg', 'jpg')
        try:
            return analyze_bytes(data, fmt)
        except RuntimeError as e:
            raise HTTPException(502, str(e))

    def _decode(item):
        import base64, binascii
        b64 = str(item.get('image', ''))
        if ',' in b64[:100]:
            b64 = b64.split(',', 1)[1]
        try:
            data = base64.b64decode(b64)
        except (binascii.Error, ValueError):
            raise HTTPException(400, 'base64 이미지가 올바르지 않습니다.')
        if not data:
            raise HTTPException(400, '빈 이미지입니다.')
        return data, str(item.get('format', 'jpg')).lower().replace('jpeg', 'jpg')

    @app.post('/analyze_multi_base64')
    def analyze_multi_base64(payload: dict):
        """사진 여러 장(앞면·뒷면): {"images": [{"image": "<base64>", "format": "jpg"}, ...]} (최대 4장)"""
        items = payload.get('images') or []
        if not items:
            raise HTTPException(400, '사진이 없습니다.')
        if len(items) > 4:
            raise HTTPException(400, '사진은 한 번에 4장까지 보낼 수 있습니다.')
        try:
            return analyze_many_bytes([_decode(x) for x in items])
        except RuntimeError as e:
            raise HTTPException(502, str(e))

    @app.post('/analyze_text')
    def analyze_text_only(payload: dict):
        """OCR 없이 글만 시험: {"text": "..."} (CLOVA 요금 없음)"""
        if isinstance(payload.get('texts'), list):          # 여러 장 시험: {"texts": ["앞면 글", "뒷면 글"]}
            return analyze_ocr_texts(payload['texts'])
        return analyze_ocr_text(payload.get('text', ''))
except ImportError:
    app = None   # fastapi 미설치 시에도 함수는 import 해서 쓸 수 있음
