// 광고 문구 분석 테스트 앱 (React Native + Expo)
// 사진 여러 장(앞면·뒷면) 찍기/고르기 → 우리 서버(/analyze_multi_base64)로 전송 → 서버가 CLOVA OCR + 분석 → 결과 표시
// CLOVA 키는 앱에 넣지 않습니다(서버 환경변수에만).
//
// 만들기:  npx create-expo-app@latest ad-test-app --template blank
//          cd ad-test-app && npx expo install expo-image-picker
//          이 파일로 App.js 교체 → npx expo start → 휴대폰 Expo Go로 QR 스캔
import { useState } from 'react';
import {
  ActivityIndicator, Alert, Image, Pressable, SafeAreaView, ScrollView,
  StyleSheet, Text, TextInput, View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

// 노트북에서 서버를 띄웠다면 노트북 IP (예: http://192.168.0.12:8000)
const DEFAULT_SERVER = 'http://10.200.241.28:8000';   // 노트북 IP가 바뀌면 여기 또는 앱 화면에서 수정

const TYPE_COLOR = {
  '질병 예방·치료 표현': '#fde2e2',
  '의약품 연상 표현': '#ffe8cc',
  '기능·효과 표현': '#fff3bf',
  '공식 기능성 문구': '#d3f9d8',
  '확인 필요': '#e9ecef',
  '해당 없음': '#f8f9fa',
};

const MAX_PHOTOS = 4;

export default function App() {
  const [server, setServer] = useState(DEFAULT_SERVER);
  const [photos, setPhotos] = useState([]);          // 분석할 사진들 (앞면, 뒷면 …)
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [showNone, setShowNone] = useState(false);
  const [showOcr, setShowOcr] = useState(false);

  async function addPhotos(fromCamera) {
    if (photos.length >= MAX_PHOTOS) {
      Alert.alert('사진 개수', `사진은 ${MAX_PHOTOS}장까지 넣을 수 있어요.`);
      return;
    }
    if (fromCamera) {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('권한 필요', '사진을 찍으려면 카메라 권한이 필요합니다.');
        return;
      }
    }
    // base64: true → 사진을 글자(base64)로 받아서 JSON으로 서버에 보냄
    // quality 0.7: 용량을 줄여 전송을 빠르게 (글자 인식에는 충분)
    const left = MAX_PHOTOS - photos.length;
    const options = { mediaTypes: ['images'], quality: 0.7, base64: true };
    const res = fromCamera
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync({ ...options, allowsMultipleSelection: true, selectionLimit: left });
    if (res.canceled) return;
    const added = res.assets.filter((a) => a.base64).slice(0, left);
    setPhotos([...photos, ...added]);
    setResult(null);
    setError(null);
  }

  function removePhoto(index) {
    setPhotos(photos.filter((_, i) => i !== index));
    setResult(null);
  }

  function reset() {
    setPhotos([]);
    setResult(null);
    setError(null);
    setShowNone(false);
    setShowOcr(false);
  }

  async function analyze() {
    if (photos.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const images = photos.map((a) => {
        const mime = a.mimeType || 'image/jpeg';
        return { image: a.base64, format: mime.includes('png') ? 'png' : 'jpg' };
      });
      // 최신 Expo의 fetch는 FormData 파일 업로드를 지원하지 않음 → base64 JSON으로 보냄
      const resp = await fetch(`${server.trim()}/analyze_multi_base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images }),
      });
      const text = await resp.text();
      if (!resp.ok) throw new Error(`서버 오류 ${resp.status}: ${text}`);
      setResult(JSON.parse(text));
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  const sentences = result ? result.sentences : [];
  // '해당 없음'이어도 규칙 안내(최초 표방, 건강기능식품 문구 등)가 있으면 보여 줌
  const shown = showNone ? sentences : sentences.filter((s) => s.type !== '해당 없음' || s.rule_tags.length > 0);
  const hiddenCount = sentences.length - shown.length;

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>광고 문구 분석 테스트</Text>

        <Text style={styles.label}>서버 주소</Text>
        <TextInput
          style={styles.input}
          value={server}
          onChangeText={setServer}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
        />

        <Text style={styles.hint}>제품 앞면과 뒷면(품목보고번호·기능성 부분)을 함께 넣으면 더 정확해요.</Text>
        <View style={styles.row}>
          <Pressable style={[styles.button, styles.secondary]} disabled={loading} onPress={() => addPhotos(true)}>
            <Text style={styles.buttonText}>사진 찍기</Text>
          </Pressable>
          <Pressable style={[styles.button, styles.secondary]} disabled={loading} onPress={() => addPhotos(false)}>
            <Text style={styles.buttonText}>앨범에서</Text>
          </Pressable>
        </View>

        {photos.length > 0 && (
          <ScrollView horizontal style={{ marginTop: 12 }}>
            {photos.map((a, i) => (
              <View key={i} style={styles.thumbBox}>
                <Image source={{ uri: a.uri }} style={styles.thumb} />
                <Text style={styles.thumbLabel}>사진 {i + 1}</Text>
                <Pressable style={styles.thumbX} onPress={() => removePhoto(i)} disabled={loading}>
                  <Text style={{ color: '#fff', fontWeight: '700' }}>✕</Text>
                </Pressable>
              </View>
            ))}
          </ScrollView>
        )}

        {photos.length > 0 && (
          <View style={[styles.row, { marginTop: 12 }]}>
            <Pressable style={[styles.button, styles.primary]} disabled={loading} onPress={analyze}>
              <Text style={styles.buttonTextPrimary}>사진 {photos.length}장 분석하기</Text>
            </Pressable>
            <Pressable style={[styles.button, styles.secondary, { flex: 0.5 }]} disabled={loading} onPress={reset}>
              <Text style={styles.buttonText}>처음부터</Text>
            </Pressable>
          </View>
        )}

        {loading && <ActivityIndicator size="large" style={{ marginVertical: 24 }} />}
        {error && <Text style={styles.error}>{error}</Text>}

        {result && (
          <View>
            {result.headline && <Text style={styles.headline}>{result.headline}</Text>}
            <ProductCard product={result.product} labels={result.labels || {}} />
            <IngredientCard items={result.ingredients || []} productKnown={result.product['분류'] === '건강기능식품'}
              labels={result.labels || {}} />

            <Text style={styles.summary}>분석 시간 {result.elapsed_sec}초</Text>

            {shown.map((s, i) => <SentenceCard key={i} s={s} />)}

            {hiddenCount > 0 && (
              <Pressable onPress={() => setShowNone(true)}>
                <Text style={styles.link}>효능과 관계없는 문장 {hiddenCount}개 보기</Text>
              </Pressable>
            )}

            <Pressable onPress={() => setShowOcr(!showOcr)}>
              <Text style={styles.link}>{showOcr ? 'OCR 원문 접기' : 'OCR 원문 보기'}</Text>
            </Pressable>
            {showOcr && <Text style={styles.ocr}>{result.ocr_text}</Text>}

            <Text style={styles.notice}>{result.notice}</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function ProductCard({ product, labels }) {
  const keys = ['제품명', '업체', '품목보고번호', '식품유형', '기능성', '확인 방법'];
  const others = product['다른 후보'] || [];
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>제품 확인: {product['분류']}</Text>
      {product['안내'] && <Text style={styles.guide}>{product['안내']}</Text>}
      {keys.filter((k) => product[k]).map((k) => (
        <Text key={k} style={styles.cardRow}>
          <Text style={{ fontWeight: '600' }}>{k}: </Text>{product[k]}
        </Text>
      ))}
      {others.length > 0 && (
        <Text style={styles.cardRow}>
          <Text style={{ fontWeight: '600' }}>다른 후보: </Text>
          {others.map((o) => `${o['제품명']}(${o['일치도']})`).join(', ')}
        </Text>
      )}
      {product['섭취시주의사항'] && (
        <View style={styles.caution}>
          <Text style={{ fontWeight: '600' }}>{labels.caution_title || '섭취 시 주의사항'}</Text>
          <Text style={styles.small}>{product['섭취시주의사항']}</Text>
        </View>
      )}
    </View>
  );
}

function IngredientCard({ items, productKnown, labels }) {
  if (items.length === 0) return null;
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>{labels.ingredient_card_title || '광고에 나온 원료와 인정 기능성'}</Text>
      {items.map((g, i) => (
        <View key={i} style={{ marginTop: 8 }}>
          <Text style={{ fontWeight: '600' }}>
            {g.short} <Text style={styles.tag}>{g.kind}</Text>
          </Text>
          {g.functions.map((f, j) => <Text key={j} style={styles.small}>· {f}</Text>)}
          {g.ad_mentions.length > 0 && (
            <Text style={styles.small}>광고에서 말한 것 중 이 원료 기능성에 있는 것: {g.ad_mentions.join(', ')}</Text>
          )}
          {productKnown && g.in_product && (
            <Text style={styles.small}>
              이 제품 인정 기능성에도 있음: {g.in_product.length > 0 ? g.in_product.join(', ') : '없음'}
            </Text>
          )}
        </View>
      ))}
      <Text style={[styles.small, { color: '#868e96', marginTop: 8 }]}>
        {labels.ingredient_card_note || ''}
      </Text>
    </View>
  );
}

function SentenceCard({ s }) {
  return (
    <View style={[styles.sentence, { backgroundColor: TYPE_COLOR[s.type] || '#f8f9fa' }]}>
      <Text style={styles.sentenceText}>{s.text}</Text>
      {s.photo && <Text style={styles.photoTag}>사진 {s.photo}</Text>}
      <Text style={styles.sentenceType}>{s.type} (확률 {s.prob})</Text>
      {s.official && (
        <Text style={styles.small}>
          공식 기능성 비교: {s.official.level}
          {s.official.targets && s.official.targets.length > 0
            ? `  (${s.official.targets.map((t) => `${t.target} ${t.found ? '○' : '✕'}`).join(', ')})`
            : ''}
        </Text>
      )}
      {s.messages.map((m, i) => <Text key={i} style={styles.small}>· {m}</Text>)}
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#fff' },
  container: { padding: 16, paddingBottom: 48 },
  title: { fontSize: 22, fontWeight: '700', marginBottom: 16, marginTop: 8 },
  label: { fontSize: 13, color: '#555', marginBottom: 4 },
  input: { borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 10, marginBottom: 12 },
  row: { flexDirection: 'row', gap: 8 },
  button: { flex: 1, paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  primary: { backgroundColor: '#0c8599' },
  secondary: { borderWidth: 1, borderColor: '#0c8599' },
  buttonTextPrimary: { color: '#fff', fontWeight: '600', fontSize: 16 },
  buttonText: { color: '#0c8599', fontWeight: '600', fontSize: 16 },
  preview: { width: '100%', height: 220, marginTop: 12, backgroundColor: '#f1f3f5' },
  error: { color: '#c92a2a', marginTop: 12 },
  card: { borderWidth: 1, borderColor: '#dee2e6', borderRadius: 10, padding: 12, marginTop: 16 },
  cardTitle: { fontSize: 16, fontWeight: '700', marginBottom: 4 },
  cardRow: { marginTop: 4, lineHeight: 20 },
  hint: { color: '#495057', marginBottom: 8, lineHeight: 20 },
  thumbBox: { marginRight: 10, alignItems: 'center' },
  thumb: { width: 96, height: 128, borderRadius: 8, backgroundColor: '#f1f3f5' },
  thumbLabel: { fontSize: 12, marginTop: 4, color: '#495057' },
  thumbX: { position: 'absolute', top: 4, right: 4, width: 24, height: 24, borderRadius: 12,
            backgroundColor: 'rgba(0,0,0,0.6)', alignItems: 'center', justifyContent: 'center' },
  photoTag: { fontSize: 12, color: '#868e96', marginTop: 2 },
  headline: { marginTop: 16, fontSize: 15, fontWeight: '600', lineHeight: 22 },
  guide: { marginTop: 2, marginBottom: 4, color: '#495057', lineHeight: 20 },
  caution: { marginTop: 10, padding: 10, borderRadius: 8, backgroundColor: '#fff9db' },
  tag: { fontSize: 12, color: '#868e96', fontWeight: '400' },
  summary: { marginTop: 12, marginBottom: 4, color: '#495057' },
  sentence: { borderRadius: 8, padding: 10, marginTop: 8 },
  sentenceText: { fontSize: 15 },
  sentenceType: { fontWeight: '700', marginTop: 4 },
  small: { fontSize: 13, marginTop: 2, lineHeight: 19 },
  link: { color: '#0c8599', marginTop: 12, fontWeight: '600' },
  ocr: { marginTop: 8, color: '#495057', lineHeight: 20 },
  notice: { marginTop: 16, color: '#868e96', fontSize: 12 },
});
