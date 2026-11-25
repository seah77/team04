# -*- coding: utf-8 -*-
"""
광고 글에 나온 원료(유산균, 홍삼, 밀크씨슬, NVP-1702 ...)를 찾아 그 원료의 인정 기능성을 돌려줌.
표: data/ingredients/ingredient_functions.json  (tools/build_ingredients.py 로 만듦)

※ '건강기능식품 원료로서의 인정 기능성'입니다. 제품마다 신고한 기능성은 다를 수 있고,
   일반식품에 들어 있으면 이 기능성을 그대로 표시할 수 없습니다.
"""
import json, os, re, sys

SRC = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(os.path.dirname(SRC), 'data', 'ingredients', 'ingredient_functions.json')
sys.path.insert(0, SRC)
import ad_rules as R

_CODE_RX = re.compile(r'(?<![A-Za-z0-9])([A-Za-z]{2,6})[- ]?(\d{2,6})([A-Za-z]?)(?![A-Za-z0-9])')
_DATA = None


def _load():
    global _DATA
    if _DATA is None:
        try:
            d = json.load(open(TABLE, encoding='utf-8'))
        except FileNotFoundError:
            d = {'goshi': [], 'individual': []}
        for g in d['goshi']:
            g['rx'] = re.compile(g['pattern'])
            g['fn_targets'] = [set(R.targets(f)) for f in g['functions']]
        code_map = {}
        for x in d['individual']:
            x['fn_targets'] = [set(R.targets(f)) for f in x['functions']]
            for c in x['codes']:
                if len(c) >= 5:                    # 짧은 코드(OS20 등)는 우연히 맞을 수 있어 제외
                    code_map[c] = x
        d['code_map'] = code_map
        _DATA = d
    return _DATA


def find_ingredients(text):
    """광고 글 → [{'name', 'kind', 'matched', 'functions', 'targets'}]"""
    d = _load(); t = str(text); out = []
    for g in d['goshi']:
        m = g['rx'].search(t)
        if m:
            out.append({'name': g['name'], 'short': g['name'], 'kind': '고시형', 'matched': m.group(0),
                        'functions': g['functions'],
                        'targets': sorted(set().union(*g['fn_targets'])), '_fn_targets': g['fn_targets']})
    seen = set()
    for m in _CODE_RX.finditer(t):
        c = ''.join(m.groups()).upper()
        x = d['code_map'].get(c)
        if x and id(x) not in seen:
            seen.add(id(x))
            out.append({'name': x['name'], 'short': f"{m.group(0)} (개별인정형)", 'kind': '개별인정형',
                        'matched': m.group(0), 'functions': x['functions'],
                        'targets': sorted(set().union(*x['fn_targets'])), '_fn_targets': x['fn_targets'],
                        'recog_no': x.get('recog_no', '')})
    return out


def functions_for(target, ingredients):
    """대상어(예: '장')가 들어 있는 원료 기능성 [(원료 이름, 기능성 문구)]"""
    hits = []
    for g in ingredients:
        fs = [f for f, tg in zip(g['functions'], g['_fn_targets']) if target in tg]
        if fs:   # 대상어가 글자 그대로 들어 있는 문구를 우선 ('장' → '장 건강에 도움을 줄 수 있음')
            fs.sort(key=lambda f: 0 if re.search(rf'(?<![가-힣]){re.escape(target)}', f) else 1)
            hits.append((g['short'], fs[0]))
    return hits


def public(ingredients):
    """앱으로 보낼 때 내부용 필드 제거"""
    return [{k: v for k, v in g.items() if not k.startswith('_')} for g in ingredients]
