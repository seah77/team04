# -*- coding: utf-8 -*-
"""
사진(광고·포장·전단지·캡처) → 문장 목록 엑셀 자동 작성

  python src/ocr_to_sheet.py 사진폴더                 # EasyOCR로 읽기 (Colab은 GPU 런타임 권장)
  python src/ocr_to_sheet.py 사진폴더 --test_ratio 0.3  # 사진의 30%를 시험셋으로 표시

- 폴더 안의 .jpg/.jpeg/.png/.webp 는 EasyOCR로 읽고, .txt 는 ML Kit 등에서 이미 읽은 글로 보고 그대로 씁니다.
- 결과: data/labels/collected_날짜.xlsx (data/templates/collect_template.xlsx 형식, 드롭다운·라벨 기준 시트 포함)
- models/ad_model_v2.joblib 가 있으면 '초안 라벨'을 자동으로 채웁니다(참고용). 라벨러1·2 칸은 사람이 채웁니다.
- OCR 오타는 고치지 마세요. 실제 앱에서도 그렇게 들어옵니다. 뜻을 알 수 없는 줄만 지우면 됩니다.
"""
import argparse, glob, os, random, re, sys, time
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import ad_rules as R

IMG = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def group_lines(results, min_conf=0.2):
    """EasyOCR 결과(상자, 글자, 신뢰도) → 위에서 아래로, 같은 줄끼리 이어 붙인 줄 목록"""
    items = []
    for box, text, conf in results:
        if conf < min_conf or not text.strip():
            continue
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        items.append({'y': (min(ys) + max(ys)) / 2, 'h': max(ys) - min(ys), 'x': min(xs), 't': text.strip()})
    if not items:
        return []
    items.sort(key=lambda d: d['y'])
    hs = sorted(d['h'] for d in items); mh = hs[len(hs) // 2] or 1
    lines, cur = [], [items[0]]
    for d in items[1:]:
        if abs(d['y'] - sum(c['y'] for c in cur) / len(cur)) < 0.5 * mh:
            cur.append(d)
        else:
            lines.append(cur); cur = [d]
    lines.append(cur)
    return [' '.join(c['t'] for c in sorted(l, key=lambda c: c['x'])) for l in lines]


def read_folder(folder, use_gpu=None):
    files = sorted(f for f in glob.glob(os.path.join(folder, '*')) if f.lower().endswith(IMG + ('.txt',)))
    reader = None
    out = []
    for f in files:
        name = os.path.basename(f)
        if f.lower().endswith('.txt'):
            text = open(f, encoding='utf-8', errors='ignore').read()
        else:
            if reader is None:
                import easyocr, torch
                gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
                print(f'EasyOCR 준비 중 (GPU={gpu}) ... 처음엔 모델을 내려받아 1~2분 걸립니다')
                reader = easyocr.Reader(['ko', 'en'], gpu=gpu)
            t0 = time.time()
            text = '\n'.join(group_lines(reader.readtext(f, detail=1)))
            print(f'  {name}: {time.time() - t0:.1f}초')
        sents = R.split_sentences(text)
        out.append((name, sents))
        print(f'{name}: 문장 {len(sents)}개')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    p.add_argument('--template', default=os.path.join(ROOT, 'data', 'templates', 'collect_template.xlsx'))
    p.add_argument('--model', default=os.path.join(ROOT, 'models', 'ad_model_v2.joblib'))
    p.add_argument('--test_ratio', type=float, default=0.25, help='시험셋으로 표시할 사진 비율')
    p.add_argument('--kind', default='', help='제품 종류 일괄 입력: 건강기능식품 / 일반식품 / 모름')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    data = read_folder(a.folder)
    if not data:
        sys.exit('사진(.jpg/.png) 또는 .txt 파일을 찾지 못했습니다.')

    pred = None
    if os.path.exists(a.model):
        from train_ad_classifier_v2 import load_bundle, rule_override
        b = load_bundle(a.model)
        pred = lambda ss: [rule_override(s, l) for s, l in zip(ss, b['model'].predict(ss))] if ss else []

    names = [n for n, _ in data]
    random.Random(42).shuffle(names)
    test = set(names[:round(len(names) * a.test_ratio)])

    from openpyxl import load_workbook
    wb = load_workbook(a.template)
    ws = wb['정답표']
    if ws.max_row >= 2:
        ws.delete_rows(2, ws.max_row - 1)
    n = 0; seen = set()
    for name, sents in data:
        labs = pred(sents) if pred else [''] * len(sents)
        for s, lab in zip(sents, labs):
            k = (name, s.replace(' ', ''))
            if k in seen:
                continue
            seen.add(k); n += 1
            ws.append([n, s, lab, '', '', '', '시험' if name in test else '학습', a.kind, name, ''])
    out = a.out or os.path.join(ROOT, 'data', 'labels', f'collected_{time.strftime("%Y%m%d_%H%M")}.xlsx')
    wb.save(out)
    print(f'\n저장: {out}  (사진 {len(data)}장, 문장 {n}개, 시험셋 사진 {len(test)}장)')
    print('다음: 엑셀을 열어 라벨러1(·라벨러2) 칸을 채운 뒤 학습 폴더에 넣고 다시 학습하세요.')


if __name__ == '__main__':
    main()
