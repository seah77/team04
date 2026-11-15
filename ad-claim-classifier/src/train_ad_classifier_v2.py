# -*- coding: utf-8 -*-
"""
광고 문구 유형 분류기 v2 — 규칙 + 머신러닝

[유형 5개]  질병 예방·치료 표현 / 의약품 연상 표현 / 기능·효과 표현 / 공식 기능성 문구 / 해당 없음
[규칙 태그] 체험기·후기, 인증·보증, 원료 효능, 한약 처방명, 질병명, 의약품 (ad_rules.py)

폴더 구조 (저장소 루트 기준)
  src/ad_rules.py, src/train_ad_classifier_v2.py
  data/labels/*.xlsx        라벨 엑셀 ('정답표' 시트가 있는 파일은 모두 학습에 사용)
  data/official/            I0030.csv.gz (필수), I-0040.csv.gz, I1250_csv.gz (선택)

실행 (저장소 루트에서)
  python src/train_ad_classifier_v2.py                       # 학습 + 평가 + 저장
  python src/train_ad_classifier_v2.py --embed jhgan/ko-sroberta-multitask   # 임베딩 모델 비교
  python src/train_ad_classifier_v2.py --predict "문구1" "문구2"
  python src/train_ad_classifier_v2.py --analyze "OCR로 읽은 전체 글"
결과
  models/ad_model_v2.joblib, reports/train_report.txt, reports/misclassified.csv, reports/threshold_table.csv

※ 앱은 위반 여부를 판정하지 않습니다. 결과는 "관련될 수 있는 표현"으로만 보여줍니다.
"""
import argparse, glob, io, json, os, re, sys, time
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import ad_rules as R


def P(*parts):
    """저장소 루트 기준 경로"""
    return os.path.join(ROOT, *parts)

LABELS = ['질병 예방·치료 표현', '의약품 연상 표현', '기능·효과 표현', '공식 기능성 문구', '해당 없음']
OFFICIAL, NONE = '공식 기능성 문구', '해당 없음'
SEED = 42
VERSION = 'v2'


def norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip()


def clean_label(v):
    if not isinstance(v, str):
        return ''
    v = v.strip()
    old = {'기능성 유사 표현': '기능·효과 표현', '신체 기능·효과 표현': '기능·효과 표현',
           '기만(원료효능·체험기 등)': '기능·효과 표현'}
    v = old.get(v, v)
    return v if v in LABELS else ''


# ------------------------------------------------------------------ 데이터 읽기
def load_sheets(pattern):
    frames, conflicts, skipped = [], [], []
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path).startswith('~$') or os.path.basename(path) in ('문장수집_양식.xlsx', 'collect_template.xlsx'):
            continue
        try:
            df = pd.read_excel(path, sheet_name='정답표', dtype=str)
        except Exception:
            continue
        col = '문장' if '문장' in df.columns else ('문구' if '문구' in df.columns else None)
        if col is None:
            continue
        for _, r in df.iterrows():
            t = norm(r.get(col, ''))
            if not t or t == 'nan':
                continue
            fin, l1, l2 = (clean_label(r.get(k)) for k in ('최종 라벨', '라벨러1', '라벨러2'))
            draft = clean_label(r.get('초안 라벨')) or clean_label(r.get('우리 라벨(검수)', '').replace(' (초안)', '') if isinstance(r.get('우리 라벨(검수)'), str) else '')
            if fin:
                lab, how = fin, '최종'
            elif l1 and l2:
                if l1 == l2:
                    lab, how = l1, '두 명 일치'
                else:
                    conflicts.append((os.path.basename(path), t, l1, l2)); continue
            elif l1 or l2:
                lab, how = l1 or l2, '한 명'
            elif draft:
                lab, how = draft, '초안'
            else:
                skipped.append((os.path.basename(path), t)); continue
            split = '시험' if str(r.get('학습/시험', '')).strip() == '시험' else '학습'
            if split == '시험' and how == '초안':
                skipped.append((os.path.basename(path), t + ' (시험셋인데 사람 라벨 없음)')); continue
            frames.append({'text': t, 'label': lab, 'how': how, 'split': split,
                           'source': f"{os.path.basename(path)} | {r.get('출처', '') if isinstance(r.get('출처'), str) else ''}"})
    return pd.DataFrame(frames), conflicts, skipped


def official_pool(path):
    if not os.path.exists(path):
        return []
    d = pd.read_csv(path, usecols=['PRIMARY_FNCLTY'], dtype=str).dropna()
    pool = set()
    for t in d['PRIMARY_FNCLTY']:
        for s in re.split(r'[\n\r]+|(?<=음)\s*[,.]?\s+(?=[①-⑳\d(\-·]|\[)|(?<=[.])\s+', t):
            s = norm(re.sub(r'^[\s①-⑳\d.)\-(·\[\]]+', '', s)).strip(' ,.')
            if 6 <= len(s) <= 80 and '도움' in s:
                pool.add(s)
    return sorted(pool)


def product_names(path, n):
    if n <= 0 or not os.path.exists(path):
        return []
    d = pd.read_csv(path, usecols=['PRDLST_NM'], dtype=str, nrows=300000)['PRDLST_NM'].dropna().map(norm)
    block = '|'.join(map(re.escape, ['면역', '다이어트', '수면', '숙면', '붓기', '부기', '혈당', '당뇨', '관절', '탈모',
                                     '키성장', '키크', '피로', '약', '예방', '치료', '건강', '디톡스', '해독']))
    d = d[(d.str.len() >= 3) & (d.str.len() <= 30) & ~d.str.contains(block)].drop_duplicates()
    return d.sample(n=min(n, len(d)), random_state=SEED).tolist()


# ------------------------------------------------------------------ 모델
class EmbedFeatures(BaseEstimator, TransformerMixin):
    """한국어 문장 임베딩 (sentence-transformers 필요, Colab: pip install sentence-transformers)"""
    def __init__(self, model_name='jhgan/ko-sroberta-multitask'):
        self.model_name = model_name
        self._m = None

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name)
        return self._m

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._model().encode(list(X), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    def __getstate__(self):
        s = self.__dict__.copy(); s['_m'] = None
        return s


def make_model(kind, embed_name=None):
    tfidf = ('글자', TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), sublinear_tf=True))
    rules = ('규칙', R.RuleFeatures(weight=2.0))
    feats = {'글자만': [tfidf], '글자+규칙': [tfidf, rules],
             '임베딩+규칙': [('임베딩', EmbedFeatures(embed_name)), rules]}[kind]
    return Pipeline([('특징', FeatureUnion(feats)),
                     ('분류', LogisticRegression(max_iter=4000, class_weight='balanced', C=4.0))])


NUTRIENT = re.compile(r'^[\s\d.,/%()]*(열량|칼로리|kcal|나트륨|탄수화물|당류|지방|포화지방|트랜스지방|콜레스테롤|단백질|식이섬유|'
                      r'칼슘|철분?|비타민\s?[A-Z]?\d*|총\s?내용량|내용량|영양정보|영양성분|1일\s?영양성분\s?기준치)'
                      r'[\s\d.,/%()]*(k?cal|m?g|㎎|㎍|μg|mL|ml|%)?[\s\d.,/%()]*$', re.I)


def rule_override(text, ml_label):
    """규칙이 확실히 잡은 것은 규칙 결과를 우선"""
    if NUTRIENT.match(text.strip()):          # 영양성분표 항목(지방 3g, 콜레스테롤 0mg ...)은 효능 표현이 아님
        return NONE
    tags = {k for k, _ in R.rule_tags(text)}
    if '질병 예방·치료 표현' in tags:
        return '질병 예방·치료 표현'
    if tags & {'의약품 연상 표현', '한약 처방명'}:
        return '의약품 연상 표현'
    # 체험기·원료 효능은 효과를 말하는 문장 → '해당 없음'으로 나오면 기능·효과로 올림
    if ml_label in (NONE, '확인 필요') and tags & {'체험기·후기 표현', '원료 효능 설명'}:
        return '기능·효과 표현'
    return ml_label


# ------------------------------------------------------------------ 공식 기능성 비교
STOP = set('도움 줄 수 있음 있는 개선 건강 기능 유지 필요 증진 감소 증가 억제 촉진 보호 상승 저하 향상 완화 개선에 '
           '건강에 유지에 도움을 효과 효능 위한 및 의 에 을 를 으로 탁월 좋은'.split())


def key_tokens(s):
    toks = re.findall(r'[가-힣A-Za-z0-9]{2,}', s)
    out = set()
    for t in toks:
        t = re.sub(r'(에|의|을|를|이|가|은|는|과|와|으로|로|에도|에는)$', '', t)
        if len(t) >= 2 and t not in STOP:
            out.add(t)
    return out


class OfficialMatcher:
    def __init__(self, pool):
        self.pool = pool
        self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True)
        self.M = self.vec.fit_transform(pool) if pool else None

    def match(self, text, candidates=None):
        pool = candidates or self.pool
        if not pool:
            return {'level': '비교 불가', 'nearest': None, 'score': 0.0}
        if candidates:
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True).fit(pool + [text])
            M, q = vec.transform(pool), vec.transform([text])
        else:
            M, q = self.M, self.vec.transform([text])
        sims = (M @ q.T).toarray().ravel()
        i = int(sims.argmax()); sc = float(sims[i]); near = pool[i]
        shared = key_tokens(text) & key_tokens(near)
        official_form = bool(re.search(r'도움을?\s*줄\s*수\s*있', text))
        if sc >= 0.7 and shared and official_form:   # 공식 문구 형식("~에 도움을 줄 수 있음")일 때만 '확인됨'
            lv = '확인됨'
        elif sc >= 0.3 and shared:
            lv = '표현이 다름'
        else:
            lv = '확인되지 않음'
        return {'level': lv, 'nearest': near if lv != '확인되지 않음' else None, 'score': round(sc, 2),
                'shared': sorted(shared)}


# ------------------------------------------------------------------ 분석 (서버에서 쓰는 함수)
def load_bundle(path):
    b = joblib.load(path)
    R.INGREDIENT[:] = b['ingredients']
    b['matcher'] = OfficialMatcher(b['official_pool'])
    return b


def analyze_text(text, bundle, product_fnclty=None, threshold=None):
    """OCR 전체 글 → 문장별 결과 목록 (FastAPI에서 그대로 JSON으로 돌려주면 됨)"""
    th = bundle['threshold'] if threshold is None else threshold
    sents = R.split_sentences(text)
    if not sents:
        return []
    proba = bundle['model'].predict_proba(sents)
    classes = list(bundle['model'].classes_)
    cand = R.split_sentences(product_fnclty) if product_fnclty else None
    out = []
    for s, pr in zip(sents, proba):
        i = int(np.argmax(pr)); p = float(pr[i])
        ml = classes[i] if p >= th else '확인 필요'
        tags = R.rule_tags(s)
        final = rule_override(s, ml)
        item = {'text': s, 'type': final, 'ml_type': classes[i], 'prob': round(p, 2),
                'rule_tags': [{'tag': k, 'evidence': v} for k, v in tags], 'messages': []}
        if final in ('기능·효과 표현', '공식 기능성 문구'):
            item['official'] = bundle['matcher'].match(s, cand)
        for k, _ in tags:
            item['messages'].append(R.DISPLAY.get(k, ''))
        if final == '기능·효과 표현' and not tags:
            item['messages'].append('몸의 기능·효과를 말하는 표현입니다. 공식 기능성 문구와 비교해 보세요.')
        if final == '확인 필요':
            item['messages'].append('표현 유형을 자동으로 판단하기 어렵습니다.')
        item['messages'] = list(dict.fromkeys(m for m in item['messages'] if m))
        out.append(item)
    return out


# ------------------------------------------------------------------ 학습
def report_block(title, y, p, labels):
    print(f'=== {title} ===')
    print(classification_report(y, p, labels=labels, digits=2, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, p, labels=labels), index=labels, columns=labels)
    print('혼동행렬 (행=정답, 열=예측)'); print(cm.to_string()); print()


def threshold_table(model, X, y):
    proba = model.predict_proba(X); cls = model.classes_
    pred = cls[proba.argmax(1)]; conf = proba.max(1)
    rows = []
    for th in [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7]:
        m = conf >= th
        acc = float((pred[m] == np.array(y)[m]).mean()) if m.any() else float('nan')
        rows.append({'기준값': th, '자동 판단 비율': round(float(m.mean()), 2), '자동 판단한 것의 정확도': round(acc, 2),
                     '확인 필요로 넘긴 수': int((~m).sum())})
    return pd.DataFrame(rows)


def train(a):
    log = io.StringIO()
    R.load_ingredients(a.i0040)
    sheets, conflicts, skipped = load_sheets(a.sheets)
    if sheets.empty:
        sys.exit('라벨 엑셀을 찾지 못했습니다. 같은 폴더에 .xlsx(정답표 시트)를 두세요.')
    pool = official_pool(a.i0030)
    rng = np.random.default_rng(SEED)
    off = [pool[i] for i in rng.choice(len(pool), size=min(a.n_official, len(pool)), replace=False)] if pool else []
    extra = pd.DataFrame([{'text': t, 'label': OFFICIAL, 'how': '공식 데이터', 'split': '학습', 'source': 'I0030'} for t in off] +
                         [{'text': t, 'label': NONE, 'how': '제품명', 'split': '학습', 'source': 'I1250 제품명'}
                          for t in product_names(a.i1250, a.n_names)])
    df = pd.concat([sheets, extra], ignore_index=True)
    df = df[df['text'].str.len() >= 2]
    df = df.sort_values('split', key=lambda c: c.ne('시험'), kind='stable').reset_index(drop=True)  # 중복이면 시험셋 쪽을 남김
    key = df['text'].str.replace(' ', '')
    multi = df.groupby(key)['label'].nunique(); multi = list(multi[multi > 1].index)
    df = df.loc[~key.duplicated(keep='first')].reset_index(drop=True)

    with redirect_stdout(Tee(log)):
        print(f'광고 문구 분류기 {VERSION}  ({time.strftime("%Y-%m-%d %H:%M")})')
        print(f'전체 {len(df)}건  (학습 {int((df.split == "학습").sum())}, 시험 {int((df.split == "시험").sum())})')
        print(df.groupby(['label', 'split']).size().unstack(fill_value=0).to_string())
        print('\n라벨 출처:', df['how'].value_counts().to_dict())
        if multi:
            print(f'[검토] 같은 문구에 서로 다른 라벨 {len(multi)}건: {multi[:10]}')
        if conflicts:
            print(f'[검토] 라벨러1·2가 다르고 최종 라벨이 없어 뺀 문장 {len(conflicts)}건 (예: {conflicts[:3]})')
        if skipped:
            print(f'[참고] 라벨이 없어 뺀 문장 {len(skipped)}건 (예: {skipped[:3]})')
        print()

        tr, te = df[df.split == '학습'], df[df.split == '시험']
        real_test = len(te) >= 20
        if not real_test:
            print('※ 사람이 라벨한 시험셋("시험")이 20건 미만이라, 학습 데이터에서 20%를 임시 시험셋으로 뗍니다.')
            print('  이 점수는 사례집·합성 문장 기준이라 실제 사진에서의 성능이 아닙니다.\n')
            tr, te = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
        labels = [l for l in LABELS if l in set(df.label)]

        kinds = ['글자만', '글자+규칙'] + (['임베딩+규칙'] if a.embed else [])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = {}
        for k in kinds:
            try:
                p = cross_val_predict(make_model(k, a.embed), tr['text'], tr['label'], cv=cv)
            except ImportError:
                print('임베딩 비교를 건너뜀: pip install sentence-transformers 가 필요합니다.'); continue
            scores[k] = f1_score(tr['label'], p, average='macro')
            print(f'[교차검증] {k:8s} macro F1 = {scores[k]:.3f}')
        best = max(scores, key=scores.get)
        print(f'→ 선택한 모델: {best}\n')

        model = make_model(best, a.embed).fit(tr['text'], tr['label'])
        te_ml = model.predict(te['text'])
        te_final = [rule_override(t, p) for t, p in zip(te['text'], te_ml)]
        report_block(f'시험셋 ({"사람이 만든 시험셋" if real_test else "임시"}, {len(te)}건) — 머신러닝만', te['label'], te_ml, labels)
        report_block('시험셋 — 규칙 우선 적용 후 (앱에서 실제로 나가는 결과)', te['label'], te_final, labels)
        f1_final = f1_score(te['label'], te_final, average='macro')
        rec = pd.Series(classification_report(te['label'], te_final, labels=labels, output_dict=True, zero_division=0))
        print(f'핵심 지표: macro F1 {f1_final:.3f} / 질병 재현율 {rec["질병 예방·치료 표현"]["recall"]:.2f}'
              + (f' / 의약품 재현율 {rec["의약품 연상 표현"]["recall"]:.2f}' if '의약품 연상 표현' in rec else ''))

        tt = threshold_table(model, te['text'], te['label'])
        print('\n기준값별 "확인 필요" 비율 (시험셋)'); print(tt.to_string(index=False))
        tt.to_csv(P('reports', 'threshold_table.csv'), index=False, encoding='utf-8-sig')

        wrong = te.assign(예측_머신러닝=te_ml, 예측_최종=te_final)
        wrong = wrong[wrong['label'] != wrong['예측_최종']]
        wrong['규칙 태그'] = wrong['text'].map(lambda t: ', '.join(k for k, _ in R.rule_tags(t)))
        wrong[['text', 'label', '예측_최종', '예측_머신러닝', '규칙 태그', 'source']].rename(columns={'text': '문장', 'label': '정답'}) \
            .to_csv(P('reports', 'misclassified.csv'), index=False, encoding='utf-8-sig')
        print(f'\n틀린 문장 {len(wrong)}건 → reports/misclassified.csv')

        final = make_model(best, a.embed).fit(df['text'], df['label'])
        bundle = {'model': final, 'kind': best, 'labels': labels, 'threshold': a.threshold, 'version': VERSION,
                  'official_pool': pool, 'ingredients': list(R.INGREDIENT),
                  'scores': {'cv_macro_f1': scores, 'test_macro_f1': round(f1_final, 3), 'real_test': real_test,
                             'n_train': int(len(tr)), 'n_test': int(len(te))}}
        joblib.dump(bundle, a.out)
        print(f'모델 저장: {a.out}  (전체 {len(df)}건으로 다시 학습)\n')

        demo = '면역력 증진에 탁월한 6년근 홍삼\n당뇨약 대신 드세요\n혈행 개선에 도움을 줄 수 있음\n먹기만 해도 붓기가 쏙\n' \
               '3개월 먹었더니 키가 5cm 컸어요\n고소한 옥수수 수염차\n치매 걱정 끝\n식약처 인증 완료\n체온이 올라가요'
        show_analysis(load_bundle(a.out), demo)
    open(P('reports', 'train_report.txt'), 'w', encoding='utf-8').write(log.getvalue())
    print('보고서 저장: reports/train_report.txt')


class Tee(io.TextIOBase):
    def __init__(self, buf):
        self.buf = buf

    def write(self, s):
        sys.__stdout__.write(s); self.buf.write(s); return len(s)


def show_analysis(bundle, text, fn=None):
    print('=== 분석 예시 ===')
    for it in analyze_text(text, bundle, fn):
        tags = ', '.join(f"{t['tag']}({t['evidence']})" for t in it['rule_tags'])
        off = it.get('official')
        offs = f" | 공식 비교: {off['level']}" + (f" ← '{off['nearest']}' ({off['score']})" if off and off['nearest'] else '') if off else ''
        print(f"- {it['text']}\n    → {it['type']}  [머신러닝 {it['ml_type']} {it['prob']}]" + (f'  규칙: {tags}' if tags else '') + offs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sheets', default=P('data', 'labels', '*.xlsx'))
    p.add_argument('--i0030', default=P('data', 'official', 'I0030.csv.gz'))
    p.add_argument('--i0040', default=P('data', 'official', 'I-0040.csv.gz'))
    p.add_argument('--i1250', default=P('data', 'official', 'I1250_csv.gz'))
    p.add_argument('--n_official', type=int, default=150)
    p.add_argument('--n_names', type=int, default=40, help='일반식품 제품명을 해당 없음으로 넣을 수 (0이면 안 넣음)')
    p.add_argument('--threshold', type=float, default=0.45)
    p.add_argument('--embed', default=None, help='예: jhgan/ko-sroberta-multitask')
    p.add_argument('--out', default=P('models', 'ad_model_v2.joblib'))
    p.add_argument('--predict', nargs='*')
    p.add_argument('--analyze')
    p.add_argument('--fnclty', help='--analyze 와 함께: 제품의 공식 기능성 문구')
    a = p.parse_args()
    if a.predict or a.analyze:
        b = load_bundle(a.out)
        show_analysis(b, a.analyze if a.analyze else '\n'.join(a.predict), a.fnclty)
        return
    train(a)


if __name__ == '__main__':
    main()
