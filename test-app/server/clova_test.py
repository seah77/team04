# -*- coding: utf-8 -*-
"""
CLOVA OCR 시험 + 광고 문구 분석까지 한 번에 (조원 clova_test.py 개선판)

바뀐 점
  - URL·Secret Key를 코드에 쓰지 않고 환경변수(또는 .env 파일)에서 읽음 → 깃허브에 올려도 안전
  - 사진 1장 또는 폴더 통째로 처리
  - 결과 저장: ocr_out/<사진이름>.json (CLOVA 원본), ocr_out/<사진이름>.txt (줄 정리된 글)
    → .txt 폴더를 ml/ad-claim-classifier/src/ocr_to_sheet.py 에 넣으면 시험셋 엑셀이 만들어짐
  - --analyze 를 붙이면 머신러닝 분석 결과까지 출력
  - 이미 받은 .json 이 있으면 CLOVA를 다시 부르지 않음(요금 절약). 다시 부르려면 --force

준비
  pip install requests python-dotenv
  같은 폴더에 .env 파일 (깃허브에 올리지 말 것):
      CLOVA_OCR_URL=https://....apigw.ntruss.com/custom/v1/..../general
      CLOVA_OCR_SECRET=발급받은키

사용
  python clova_test.py test.jpg
  python clova_test.py 사진폴더 --analyze
"""
import argparse, base64, glob, json, os, sys, time, uuid
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, '.env'))
except ImportError:
    pass

URL = os.environ.get('CLOVA_OCR_URL', '')
SECRET = os.environ.get('CLOVA_OCR_SECRET', '')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.pdf', '.tif', '.tiff')


def call_clova(path, timeout=30):
    """사진 파일 → CLOVA General OCR 응답(dict). base64 JSON 방식(조원 코드와 같은 방식)."""
    if not URL or not SECRET:
        sys.exit('CLOVA_OCR_URL / CLOVA_OCR_SECRET 을 .env 또는 환경변수에 넣어 주세요.')
    fmt = os.path.splitext(path)[1].lstrip('.').lower()
    fmt = 'jpg' if fmt == 'jpeg' else fmt
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    body = {'version': 'V2', 'requestId': str(uuid.uuid4()), 'timestamp': int(time.time() * 1000),
            'lang': 'ko', 'images': [{'format': fmt, 'name': 'img', 'data': data}]}
    r = requests.post(URL, json=body, headers={'X-OCR-SECRET': SECRET}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f'CLOVA 오류 {r.status_code}: {r.text[:300]}')
    return r.json()


def lines_with_conf(resp):
    """줄 단위 (평균 신뢰도, 줄 글자) — 원래 코드는 줄의 마지막 조각 신뢰도만 찍었음"""
    out, words, confs = [], [], []
    for f in resp['images'][0].get('fields', []):
        words.append(f['inferText']); confs.append(f.get('inferConfidence', 1.0))
        if f.get('lineBreak'):
            out.append((sum(confs) / len(confs), ' '.join(words))); words, confs = [], []
    if words:
        out.append((sum(confs) / len(confs), ' '.join(words)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('target', nargs='?', default='test.jpg', help='사진 파일 또는 폴더')
    p.add_argument('--out', default='ocr_out')
    p.add_argument('--analyze', action='store_true', help='머신러닝 분석 결과도 출력')
    p.add_argument('--force', action='store_true', help='저장된 결과가 있어도 CLOVA 다시 호출')
    _cands = [os.path.join(os.path.dirname(HERE), 'ml', 'ad-claim-classifier', 'src'),            # 테스트앱 묶음
              os.path.join(os.path.dirname(os.path.dirname(HERE)), 'ad-claim-classifier', 'src')]   # 깃허브 저장소
    p.add_argument('--ml_src', default=next((c for c in _cands if os.path.isdir(c)), _cands[0]))
    a = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.target, '*')) if f.lower().endswith(IMG_EXT)) \
        if os.path.isdir(a.target) else [a.target]
    if not files:
        sys.exit('사진을 찾지 못했습니다.')
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, a.ml_src)
    from ocr_adapters import clova_to_text   # 줄 정리(신뢰도 0.5 미만 조각 제거)
    analyze = None
    if a.analyze:
        from ocr_adapters import analyze_image_text as analyze

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        jpath, tpath = os.path.join(a.out, stem + '.json'), os.path.join(a.out, stem + '.txt')
        if os.path.exists(jpath) and not a.force:
            resp = json.load(open(jpath, encoding='utf-8')); src = '저장된 결과'
        else:
            t0 = time.time(); resp = call_clova(path); src = f'CLOVA {time.time() - t0:.1f}초'
            json.dump(resp, open(jpath, 'w', encoding='utf-8'), ensure_ascii=False)
        text = clova_to_text(resp)
        open(tpath, 'w', encoding='utf-8').write(text)

        print(f'\n===== {os.path.basename(path)} ({src}) =====')
        for c, line in lines_with_conf(resp):
            print(f'{c:.2f}  {line}')
        if analyze:
            print('----- 분석 -----')
            for r in analyze(text):
                if r['type'] != '해당 없음':
                    print(f"[{r['type']}] {r['text']}  ({r['prob']})")
                    for m in r['messages']:
                        print(f'    · {m}')
            n_none = sum(r['type'] == '해당 없음' for r in analyze(text))
            print(f'(효능과 관계없는 문장 {n_none}개 생략)')
    print(f'\n저장 위치: {os.path.abspath(a.out)}  (.json 원본, .txt 줄 정리 글)')


if __name__ == '__main__':
    main()
