# -*- coding: utf-8 -*-
"""
빠른 동작 확인: python tests/check_examples.py
- 예제 OCR 결과 2개(효능 문구 없음)는 모두 '해당 없음'이어야 함
- 대표 문구가 기대 유형으로 나오는지 확인
※ 이 예제들은 규칙을 고칠 때 이미 본 문장이라 성능 점수가 아니라 '고장 났는지' 확인용입니다.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from ocr_adapters import analyze_image_text, scored_lines_to_text

ok = True


def check(name, cond, detail=''):
    global ok
    print(('통과 ' if cond else '실패 ') + name + (f'  {detail}' if detail and not cond else ''))
    ok &= cond


for f in ['ocr_nutrition_label.tsv', 'clova_monster.tsv']:
    rows = [l.split('\t') for l in open(os.path.join(ROOT, 'examples', f), encoding='utf-8').read().strip().split('\n')]
    res = analyze_image_text(scored_lines_to_text([(t, c) for c, t in rows]))
    bad = [(r['text'], r['type']) for r in res if r['type'] != '해당 없음']
    check(f'{f}: 전부 해당 없음', not bad, str(bad))

CASES = {
    '치매 걱정 끝': '질병 예방·치료 표현',
    '변비에 좋은 차': '질병 예방·치료 표현',
    '당뇨약 대신 드세요': '의약품 연상 표현',
    '부작용 없는 마시는 위고비': '의약품 연상 표현',
    '면역력 쑥쑥!': '기능·효과 표현',
    '먹기만 해도 붓기가 쏙': '기능·효과 표현',
    '혈행 개선에 도움을 줄 수 있음': '공식 기능성 문구',
    '고소하고 담백한 맛': '해당 없음',
    '포화지방 0g': '해당 없음',
}
for t, want in CASES.items():
    got = analyze_image_text(t)[0]['type']
    check(f'{t} → {want}', got == want, f'(실제: {got})')

# 제품의 인정 기능성과 대상어별 비교 (바이크롬 간 유산균 광고 배너에서 나온 사례)
FN = '(국문) 1. 비알콜성 간손상으로부터 간 건강에 도움을 줄 수 있음\n(영문) 1. May help to liver health'
r = analyze_image_text('간&장 더블케어', product_fnclty=FN, product_kind='건강기능식품')[0]
check('간&장 더블케어 → 일부만 확인(간 ○, 장 ✕)', r['official']['level'] == '일부만 확인'
      and [t['found'] for t in r['official']['targets']] == [True, False], str(r.get('official')))
r = analyze_image_text('국내 최초 간 건강 유산균')[0]
check('국내 최초 간 건강 유산균 → 기능·효과 + 최초 표방', r['type'] == '기능·효과 표현'
      and any(t['tag'] == '최초·최고 표방' for t in r['rule_tags']), str(r))
r = analyze_image_text('단절곡')[0]
check('단절곡(OCR 잡음) → 확인 필요', r['type'] == '확인 필요', r['type'])
r = analyze_image_text('붓기 쏙 빠지는 차', product_kind='일반식품')[0]
check('일반식품 + 기능 표현 → 일반식품 안내', '일반식품' in ' '.join(r['messages']), str(r['messages']))

# 원료 인식: 광고에 나온 원료의 인정 기능성으로 보충 설명
from ingredients import find_ingredients
names = [g['short'] for g in find_ingredients('바이크롬 간 유산균 NVP-1702, 밀크씨슬, 6년근 홍삼')]
check('원료 인식(유산균·NVP-1702·밀크씨슬·홍삼)', all(any(k in n for n in names) for k in ['유산균', 'NVP-1702', '밀크씨슬', '홍삼']), str(names))
r = analyze_image_text('6년근 홍삼으로 면역력 쑥쑥')[0]
check('홍삼 + 면역 → 원료 인정 기능성에 있음', any('홍삼' in m and '있음' in m for m in r['messages']), str(r['messages']))
r = analyze_image_text('간&장 더블케어 유산균', product_fnclty=FN, product_kind='건강기능식품')[0]
check('제품에 없는 장 → 유산균 일반 기능성 참고 문구', any('참고' in m and '장 건강' in m for m in r['messages']), str(r['messages']))
# 짧은 광고 구호 줄은 서로 잇지 않음
res = analyze_image_text('6년근 홍삼으로 면역력 쑥쑥\n밀크씨슬로 피로 싹\n당뇨 걱정 끝')
check('짧은 구호 3줄 → 3문장', len(res) == 3, str([x['text'] for x in res]))

# 포장 뒷면 의무 표시사항은 광고 문구로 보지 않음
for t in ['본 제품은 질병의 예방 및 치료를 위한 의약품이 아닙니다', '질환이 있거나 의약품 복용 시 전문가와 상담하십시오',
          '임산부, 수유부, 어린이는 섭취에 주의하십시오']:
    r = analyze_image_text(t)[0]
    check(f'뒷면 표시사항 → 해당 없음: {t[:15]}…', r['type'] == '해당 없음' and not r['rule_tags'], r['type'])
r = analyze_image_text('당뇨 환자도 안심하고 드세요')[0]
check('광고의 질병 표현은 그대로 잡힘', r['type'] == '질병 예방·치료 표현', r['type'])

# 안내 문구에 판정하는 말이 없는지 (data/messages.json)
import json, re as _re
_m = json.load(open(os.path.join(ROOT, 'data', 'messages.json'), encoding='utf-8'))
_all = ' '.join(v for k, sec in _m.items() if not k.startswith('_')
                for v in (sec.values() if isinstance(sec, dict) else [sec]))
_bad = _re.findall(r'불법|위반입니다|허위|과대광고|과장|효과가? 없|안전합니다|안전해요|믿을 수 있|사기', _all)
check('안내 문구에 판정 표현 없음', not _bad, str(_bad))

print('\n모두 통과' if ok else '\n실패한 항목이 있습니다')
sys.exit(0 if ok else 1)
