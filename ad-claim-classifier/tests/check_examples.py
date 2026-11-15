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

print('\n모두 통과' if ok else '\n실패한 항목이 있습니다')
sys.exit(0 if ok else 1)
