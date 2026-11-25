# -*- coding: utf-8 -*-
"""
OCR 결과(PaddleOCR / 네이버 CLOVA OCR) → 글(text) → 광고 문구 분석 연결

OCR 엔진마다 결과 모양이 다르므로, 여기서 '줄 단위 글'로 통일한 뒤
train_ad_classifier_v2.analyze_text() 에 넣습니다.

  from ocr_adapters import paddle_to_text, clova_to_text, analyze_image_text
  text = paddle_to_text(paddle_result)        # 또는 clova_to_text(clova_json)
  results = analyze_image_text(text)          # 문장별 분석 결과 (JSON으로 바로 응답 가능)

시험셋 만들기: save_text(text, '사진/IMG_0001.txt') 로 저장해 두면
  python ocr_to_sheet.py 사진   이 그 글을 읽어 문장수집 엑셀을 만듭니다(OCR 다시 안 돌림).
"""
import json, os, re, sys
SRC = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(os.path.dirname(SRC), 'models', 'ad_model_v2.joblib')
sys.path.insert(0, SRC)

MIN_CONF = 0.5   # 이보다 신뢰도가 낮은 글자 조각은 버림 (엔진·사진에 맞게 조정)


# ------------------------------------------------------------------ 공통: 상자 → 줄
def boxes_to_lines(items, min_conf=MIN_CONF):
    """items: [(네 꼭짓점 [[x,y]...], 글자, 신뢰도)] → 위→아래, 왼→오 순서의 줄 목록"""
    rows = []
    for box, text, conf in items:
        if conf is not None and conf < min_conf or not str(text).strip():
            continue
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        rows.append({'y': (min(ys) + max(ys)) / 2, 'h': max(ys) - min(ys), 'x': min(xs), 't': str(text).strip()})
    if not rows:
        return []
    rows.sort(key=lambda d: d['y'])
    hs = sorted(d['h'] for d in rows); mh = hs[len(hs) // 2] or 1
    lines, cur = [], [rows[0]]
    for d in rows[1:]:
        if abs(d['y'] - sum(c['y'] for c in cur) / len(cur)) < 0.5 * mh:
            cur.append(d)
        else:
            lines.append(cur); cur = [d]
    lines.append(cur)
    return [' '.join(c['t'] for c in sorted(l, key=lambda c: c['x'])) for l in lines]


# ------------------------------------------------------------------ PaddleOCR
def paddle_to_text(result, min_conf=MIN_CONF):
    """
    PaddleOCR 결과 → 글. 두 버전 모두 지원.
      2.x: ocr.ocr(img)      → [[ [box, (text, score)], ... ]]
      3.x: ocr.predict(img)  → [결과객체] (rec_texts, rec_scores, rec_polys 를 가진 dict 형태)
    """
    items = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue
        d = page.json['res'] if hasattr(page, 'json') and isinstance(getattr(page, 'json'), dict) and 'res' in page.json else page
        if isinstance(d, dict) and 'rec_texts' in d:                      # 3.x
            polys = d.get('rec_polys') or d.get('dt_polys') or []
            for box, t, s in zip(polys, d['rec_texts'], d.get('rec_scores', [1.0] * len(d['rec_texts']))):
                items.append(([list(map(float, p)) for p in box], t, float(s)))
        else:                                                             # 2.x
            for line in page:
                box, (t, s) = line[0], line[1]
                items.append((box, t, float(s)))
    return '\n'.join(boxes_to_lines(items, min_conf))


# ------------------------------------------------------------------ 네이버 CLOVA OCR (General)
def clova_to_text(resp, min_conf=MIN_CONF):
    """
    CLOVA OCR General 응답(JSON dict 또는 문자열) → 글.
    fields[].inferText / inferConfidence / boundingPoly.vertices / lineBreak 를 사용.
    lineBreak 가 있으면 그것으로 줄을 나누고, 없으면 좌표로 줄을 묶음.
    """
    if isinstance(resp, (str, bytes)):
        resp = json.loads(resp)
    out = []
    for img in resp.get('images', []):
        fields = img.get('fields', [])
        if fields and all('lineBreak' in f for f in fields):
            line = []
            for f in fields:
                if f.get('inferConfidence', 1.0) >= min_conf and f.get('inferText', '').strip():
                    line.append(f['inferText'].strip())
                if f.get('lineBreak'):
                    if line: out.append(' '.join(line))
                    line = []
            if line: out.append(' '.join(line))
        else:
            items = [([[v.get('x', 0), v.get('y', 0)] for v in f['boundingPoly']['vertices']],
                      f.get('inferText', ''), f.get('inferConfidence', 1.0)) for f in fields]
            out.extend(boxes_to_lines(items, min_conf))
    return '\n'.join(out)


# ------------------------------------------------------------------ 글자+신뢰도 목록 (좌표 없음)
def scored_lines_to_text(lines, min_conf=MIN_CONF):
    """
    좌표 없이 (글자, 신뢰도) 목록만 있을 때. 신뢰도는 0~1 또는 0~100 모두 가능.
    예: [('면역력 쑥쑥', 0.98), ('ㅁㄴ', 0.21)]
    ※ 순서를 맞출 수 없으므로 가능하면 좌표가 있는 원본 결과(paddle_to_text/clova_to_text)를 쓰세요.
    """
    out = []
    for t, c in lines:
        c = float(c); c = c / 100 if c > 1 else c
        if c >= min_conf and str(t).strip():
            out.append(str(t).strip())
    return '\n'.join(out)


# ------------------------------------------------------------------ 분석 연결
_BUNDLE = None


def get_bundle(model_path=DEFAULT_MODEL):
    global _BUNDLE
    if _BUNDLE is None:
        from train_ad_classifier_v2 import load_bundle
        _BUNDLE = load_bundle(model_path)
    return _BUNDLE


_END = re.compile(r'([.!?。~]|[다요죠까네음함임됨오]|\])["\')\]]*$')   # '…하십시오', '[주의사항]' 도 끝으로 봄


_LABEL = re.compile(r'품목\s*(보고|제조)?\s*번호|원재료|내용량|영양|유통기한|소비기한|제조원|판매원|섭취\s*방법|보관')


# 앞 줄이 이런 글자로 끝나면 문장이 이어지는 중으로 봄 (조사·연결어미·쉼표)
_CONT = re.compile(r'(의|및|와|과|을|를|에|에서|으로|로|고|며|서|는|은|이|가|한|할|된|될|인|하여|하고|,|·)$')


def merge_lines(text):
    """
    줄바꿈으로 잘린 문장을 다시 이어 붙임.
    앞 줄이 문장 끝(다/요/죠/!/. 등)으로 끝나지 않고 12자 이상(문단이 줄바꿈된 경우)이면 다음 줄과 이어 붙임.
    짧은 문구(광고 구호)와 표시사항 줄(품목보고번호, 원재료, 영양 등)은 잇지 않음.
    해시태그(#), 짧은 줄(제품명·단위 등)은 그대로 둠.
    """
    out = []
    for line in [l.strip() for l in str(text).split('\n') if l.strip()]:
        if (out and not _END.search(out[-1]) and not out[-1].startswith('#') and not line.startswith(('#', '[', '<', '※'))
                and len(out[-1]) >= 12 and len(line) >= 4
                and (len(out[-1]) >= 20 or _CONT.search(out[-1]))      # 짧은 구호 줄끼리는 잇지 않음
                and not _LABEL.search(out[-1]) and not _LABEL.search(line)):
            out[-1] = out[-1] + ' ' + line
        else:
            out.append(line)
    return '\n'.join(out)


def analyze_image_text(text, product_fnclty=None, model_path=DEFAULT_MODEL, merge=True, product_kind=None,
                       context_text=None):
    """OCR 글 → 문장별 분석 결과 목록 (dict 리스트, json.dumps 로 바로 응답 가능)
    product_kind: '건강기능식품' / '일반식품' / None(제품 미확인) — 안내 문구가 달라짐"""
    from train_ad_classifier_v2 import analyze_text
    return analyze_text(merge_lines(text) if merge else text, get_bundle(model_path), product_fnclty,
                        product_kind=product_kind, context_text=context_text)


def save_text(text, path):
    """시험셋용: OCR 글을 .txt로 저장 → ocr_to_sheet.py 가 그대로 읽음"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    # 사용 예: python src/ocr_adapters.py clova_응답.json
    #         python src/ocr_adapters.py examples/clova_monster.tsv   (한 줄에 '신뢰도<TAB>글자')
    p = sys.argv[1]
    raw = open(p, encoding='utf-8').read()
    if p.endswith('.json'):
        text = clova_to_text(raw)
    elif p.endswith('.tsv'):
        text = scored_lines_to_text([(l.split('\t')[1], l.split('\t')[0]) for l in raw.strip().split('\n') if '\t' in l])
    else:
        text = raw
    print('--- OCR 글 ---\n' + text + '\n--- 분석 ---')
    for r in analyze_image_text(text):
        print(f"{r['text']}  →  {r['type']} ({r['prob']})  " + ' / '.join(r['messages']))
