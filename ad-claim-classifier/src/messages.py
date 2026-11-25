"""
앱에 보이는 안내 문구 불러오기 (data/messages.json)

    from messages import msg, josa
    msg('official.hf_partial', found='간은', missing='장은', official='…')

문구를 바꿀 때는 이 파일이 아니라 data/messages.json 만 고치면 됩니다.
"""
import json
import os

PATH = os.environ.get('AD_MESSAGES', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  'data', 'messages.json'))
_DATA = None


def _load():
    global _DATA
    if _DATA is None:
        with open(PATH, encoding='utf-8') as f:
            _DATA = json.load(f)
    return _DATA


def reload():
    """서버를 끄지 않고 문구 파일을 다시 읽을 때"""
    global _DATA
    _DATA = None
    return _load()


def msg(key, **kw):
    """'official.hf_same' 처럼 점으로 구분한 이름 → 문구. 없는 이름이면 빈 글자."""
    node = _load()
    for part in key.split('.', 1) if '.' in key else [key]:
        if not isinstance(node, dict) or part not in node:
            return ''
        node = node[part]
    if not isinstance(node, str):
        return ''
    try:
        return node.format(**kw)
    except (KeyError, IndexError):
        return node


def josa(word, a='은', b='는'):
    """받침에 맞는 조사를 붙여 돌려줌: josa('간') → '간은', josa('피부') → '피부는'"""
    ch = word[-1] if word else ''
    if '가' <= ch <= '힣':
        return word + (a if (ord(ch) - 0xAC00) % 28 else b)
    return word + a


def josa_list(words, a='은', b='는'):
    """['간', '장'] → '간, 장은'"""
    return josa(', '.join(words), a, b) if words else ''
