import os
import sys
import uuid
import tempfile
import threading
import time
import platform
import json
import urllib.request
import urllib.error
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB上限

# ─── 設定 ───
ALLOWED_AUDIO = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.wma', '.aac'}
ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO | ALLOWED_VIDEO

# モデルサイズ一覧
WHISPER_MODELS = ['tiny', 'base', 'small', 'medium']

# MLX-Whisperモデル名（HuggingFace形式）
MLX_MODELS = {
    'tiny':   'mlx-community/whisper-tiny-mlx',
    'base':   'mlx-community/whisper-base-mlx',
    'small':  'mlx-community/whisper-small-mlx',
    'medium': 'mlx-community/whisper-medium-mlx',
}

# 処理状態管理
tasks = {}

# ─── データディレクトリ ───
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
QA_FILE = os.path.join(DATA_DIR, 'qa_data.json')
TAGS_FILE = os.path.join(DATA_DIR, 'tags.json')
SETTINGS_FILE = os.path.join(DATA_DIR, 'settings.json')
PROCESSED_FILE = os.path.join(DATA_DIR, 'processed_files.json')

# AI音声用_出力 フォルダパス
AI_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '仕事', 'AI音声用_出力')

# デフォルトタグ
DEFAULT_TAGS = [
    {'id': 'tag_default_01', 'name': '営業'},
    {'id': 'tag_default_02', 'name': 'マーケティング'},
    {'id': 'tag_default_03', 'name': '動画制作'},
    {'id': 'tag_default_04', 'name': 'デザイン'},
    {'id': 'tag_default_05', 'name': 'ディレクション'},
    {'id': 'tag_default_06', 'name': 'チーム管理'},
    {'id': 'tag_default_07', 'name': 'クライアント対応'},
    {'id': 'tag_default_08', 'name': 'ツール・技術'},
    {'id': 'tag_default_09', 'name': '戦略・方針'},
    {'id': 'tag_default_10', 'name': 'コンサル'},
    {'id': 'tag_default_11', 'name': 'その他'},
]

# ─── エンジン自動判定 ───
USE_MLX = False


def detect_engine():
    """Apple Silicon + mlx_whisper が使えるか判定"""
    global USE_MLX

    # macOS + Apple Silicon のみMLXを試す
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import mlx_whisper
            USE_MLX = True
            print("🍎 Apple Silicon 検出 → MLX-Whisper (GPU) を使用")
            return
        except ImportError:
            pass

    USE_MLX = False
    print("💻 汎用モード → faster-whisper (CPU) を使用")


detect_engine()

# ─── モデルキャッシュ ───
_model_cache = {}
_model_lock = threading.Lock()


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'ファイルサイズが大きすぎます。ファイルを分けてアップロードしてください。'}), 413


def get_faster_whisper_model(model_size):
    """faster-whisperモデルを取得（キャッシュ付き）"""
    cache_key = f"fw_{model_size}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    with _model_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

        from faster_whisper import WhisperModel

        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            device = "cpu"
            compute_type = "int8"

        print(f"📦 faster-whisper モデル '{model_size}' を読み込み中（{device}/{compute_type}）...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_cache[cache_key] = model
        print(f"✅ モデル '{model_size}' 準備完了")
        return model


def is_video(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO


def extract_audio_pyav(input_path, output_path):
    """PyAVで動画/音声から16kHz WAVを抽出"""
    import av
    output_container = av.open(output_path, mode='w')
    output_stream = output_container.add_stream('pcm_s16le', rate=16000, layout='mono')

    input_container = av.open(input_path)
    audio_stream = None
    for stream in input_container.streams:
        if stream.type == 'audio':
            audio_stream = stream
            break

    if audio_stream is None:
        input_container.close()
        output_container.close()
        raise RuntimeError("音声トラックが見つかりません")

    for frame in input_container.decode(audio_stream):
        frame.pts = None
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
    for packet in output_stream.encode():
        output_container.mux(packet)

    output_container.close()
    input_container.close()


def get_audio_duration(audio_path):
    """音声ファイルの長さ（秒）を取得"""
    try:
        import av
        container = av.open(audio_path)
        duration = 0
        for stream in container.streams.audio:
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
                break
        if duration <= 0 and container.duration:
            duration = float(container.duration) / 1000000.0
        container.close()
        return duration
    except Exception:
        return 0


def load_audio_as_numpy(audio_path, sr=16000):
    """PyAVで音声を読み込みnumpy配列として返す（MLX-Whisper用）"""
    import av
    import numpy as np

    container = av.open(audio_path)
    audio_stream = None
    for stream in container.streams:
        if stream.type == 'audio':
            audio_stream = stream
            break

    if audio_stream is None:
        container.close()
        raise RuntimeError("音声トラックが見つかりません")

    resampler = av.AudioResampler(format='s16', layout='mono', rate=sr)

    frames = []
    for frame in container.decode(audio_stream):
        resampled = resampler.resample(frame)
        for r in resampled:
            array = r.to_ndarray().flatten()
            frames.append(array)

    container.close()

    if not frames:
        raise RuntimeError("音声データが取得できません")

    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    return audio


# ─── MLX-Whisper 文字起こし ───
def transcribe_mlx(audio_path, model_size, task_id):
    """MLX-Whisperで高速文字起こし（Apple Silicon GPU使用）"""
    import mlx_whisper

    model_path = MLX_MODELS.get(model_size, MLX_MODELS['base'])
    total_duration = get_audio_duration(audio_path)

    tasks[task_id]['message'] = 'M4 GPUで文字起こし中...'

    audio_array = load_audio_as_numpy(audio_path)

    result = mlx_whisper.transcribe(
        audio_array,
        path_or_hf_repo=model_path,
        language='ja',
        verbose=False,
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    texts = []
    if 'segments' in result:
        for segment in result['segments']:
            text = segment.get('text', '').strip()
            if text:
                texts.append(text)
            end_time = segment.get('end', 0)
            if total_duration > 0 and end_time > 0:
                progress = min(int((end_time / total_duration) * 100), 99)
                tasks[task_id]['progress'] = progress
    elif 'text' in result:
        texts.append(result['text'].strip())

    return '\n'.join(texts) if texts else '（音声が検出されませんでした）'


# ─── faster-whisper 文字起こし（VADフィルター付き）───
def transcribe_faster(audio_path, model_size, task_id):
    """faster-whisperで文字起こし（VADフィルターで無音スキップ高速化）"""
    model = get_faster_whisper_model(model_size)
    total_duration = get_audio_duration(audio_path)

    tasks[task_id]['message'] = '文字起こし中（VADフィルター有効）...'

    # 動画の場合はまず音声を抽出
    ext = os.path.splitext(audio_path)[1].lower()
    input_path = audio_path
    wav_path = None

    if ext in ALLOWED_VIDEO:
        tasks[task_id]['message'] = '動画から音声を抽出中...'
        wav_path = audio_path + '_audio.wav'
        extract_audio_pyav(audio_path, wav_path)
        input_path = wav_path
        tasks[task_id]['message'] = '文字起こし中（VADフィルター有効）...'

    segments, info = model.transcribe(
        input_path,
        language='ja',
        beam_size=5,
        word_timestamps=False,
        condition_on_previous_text=True,
        vad_filter=True,              # 🔥 無音部分をスキップ（大幅高速化）
        vad_parameters=dict(
            min_silence_duration_ms=500,   # 500ms以上の無音をスキップ
            speech_pad_ms=200,             # 音声の前後に200msのパディング
        ),
    )

    texts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            texts.append(text)
        if total_duration > 0 and segment.end > 0:
            progress = min(int((segment.end / total_duration) * 100), 99)
            tasks[task_id]['progress'] = progress

    # 一時WAVファイルの削除
    if wav_path and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return '\n'.join(texts) if texts else '（音声が検出されませんでした）'


# ─── 統合文字起こし関数 ───
def transcribe_file(audio_path, model_size, task_id):
    """環境に応じて最適なエンジンで文字起こし"""
    if USE_MLX:
        return transcribe_mlx(audio_path, model_size, task_id)
    else:
        return transcribe_faster(audio_path, model_size, task_id)


def process_file(task_id, file_path, original_filename, model_size):
    """バックグラウンドで文字起こし処理"""
    try:
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['progress'] = 0
        start_time = time.time()

        engine = "MLX-Whisper (GPU)" if USE_MLX else "faster-whisper (CPU)"
        tasks[task_id]['message'] = f'音声を読み込み中... [{engine}]'

        text = transcribe_file(file_path, model_size, task_id)

        elapsed = time.time() - start_time
        if elapsed < 60:
            elapsed_str = f"{elapsed:.1f}秒"
        else:
            elapsed_str = f"{int(elapsed//60)}分{int(elapsed%60)}秒"

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = f'完了！（{elapsed_str}）'
        tasks[task_id]['result'] = text

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'エラー: {str(e)}'
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat')
def chat_page():
    return render_template('chat.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'ファイルが選択されていません'}), 400

    model_size = request.form.get('model', 'base')
    if model_size not in WHISPER_MODELS:
        model_size = 'base'

    task_ids = []
    job_list = []

    for file in files:
        if file.filename == '':
            continue
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        tmp_dir = tempfile.mkdtemp()
        safe_filename = str(uuid.uuid4()) + ext
        file_path = os.path.join(tmp_dir, safe_filename)
        file.save(file_path)

        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'queued',
            'message': '待機中...',
            'result': None,
            'filename': file.filename,
            'progress': 0
        }

        job_list.append((task_id, file_path, file.filename, model_size))
        task_ids.append({'task_id': task_id, 'filename': file.filename})

    if not task_ids:
        return jsonify({'error': '対応するファイルがありません'}), 400

    # 1つのスレッドで順番に処理
    def process_queue(jobs):
        for i, (tid, fpath, fname, msize) in enumerate(jobs):
            tasks[tid]['message'] = f'処理待ち（{i+1}/{len(jobs)}番目）...'
            process_file(tid, fpath, fname, msize)

    thread = threading.Thread(target=process_queue, args=(job_list,))
    thread.daemon = True
    thread.start()

    return jsonify({'tasks': task_ids})


@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'タスクが見つかりません'}), 404
    task = tasks[task_id]
    return jsonify({
        'status': task['status'],
        'message': task['message'],
        'result': task['result'],
        'filename': task['filename'],
        'progress': task.get('progress', 0)
    })


# ═══════════════════════════════════════════════════
# サーバー側 Q&A ストレージ（JSONファイル）
# ═══════════════════════════════════════════════════
_data_lock = threading.Lock()


def _read_json(filepath, default=None):
    if default is None:
        default = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_tags():
    tags = _read_json(TAGS_FILE)
    if not tags:
        _write_json(TAGS_FILE, DEFAULT_TAGS)
        return DEFAULT_TAGS[:]
    return tags


def _get_settings():
    return _read_json(SETTINGS_FILE, {})


def _generate_qa_id():
    import random
    import string
    ts = hex(int(time.time() * 1000))[2:]
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f'qa_{ts}_{rand}'


# ─── Q&A API エンドポイント ───
@app.route('/api/qa', methods=['GET'])
def api_get_qa():
    with _data_lock:
        data = _read_json(QA_FILE)
    return jsonify(data)


@app.route('/api/qa', methods=['POST'])
def api_add_qa():
    items = request.json
    if not isinstance(items, list):
        items = [items]
    with _data_lock:
        data = _read_json(QA_FILE)
        now = time.strftime('%Y-%m-%dT%H:%M:%S+09:00')
        new_items = []
        for item in items:
            new_item = {
                'id': _generate_qa_id(),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'tags': item.get('tags', []),
                'source': item.get('source', ''),
                'createdAt': now,
                'updatedAt': now
            }
            new_items.append(new_item)
        data = new_items + data
        _write_json(QA_FILE, data)
    return jsonify({'saved': len(new_items), 'total': len(data)})


@app.route('/api/qa/<qa_id>', methods=['PUT'])
def api_update_qa(qa_id):
    updates = request.json
    with _data_lock:
        data = _read_json(QA_FILE)
        for qa in data:
            if qa['id'] == qa_id:
                qa.update(updates)
                qa['updatedAt'] = time.strftime('%Y-%m-%dT%H:%M:%S+09:00')
                break
        _write_json(QA_FILE, data)
    return jsonify({'ok': True})


@app.route('/api/qa/<qa_id>', methods=['DELETE'])
def api_delete_qa(qa_id):
    with _data_lock:
        data = [qa for qa in _read_json(QA_FILE) if qa['id'] != qa_id]
        _write_json(QA_FILE, data)
    return jsonify({'ok': True, 'total': len(data)})


# ─── タグ API ───
@app.route('/api/tags', methods=['GET'])
def api_get_tags():
    with _data_lock:
        return jsonify(_get_tags())


@app.route('/api/tags', methods=['POST'])
def api_add_tag():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({'error': 'タグ名が必要です'}), 400
    with _data_lock:
        tags = _get_tags()
        existing = next((t for t in tags if t['name'] == name), None)
        if existing:
            return jsonify(existing)
        new_tag = {'id': f'tag_{uuid.uuid4().hex[:8]}', 'name': name}
        tags.append(new_tag)
        _write_json(TAGS_FILE, tags)
    return jsonify(new_tag)


@app.route('/api/tags/<tag_id>', methods=['DELETE'])
def api_delete_tag(tag_id):
    with _data_lock:
        tags = [t for t in _get_tags() if t['id'] != tag_id]
        _write_json(TAGS_FILE, tags)
    return jsonify({'ok': True})


# ─── 設定 API ───
@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    with _data_lock:
        return jsonify(_get_settings())


@app.route('/api/settings', methods=['POST'])
def api_save_settings():
    updates = request.json
    with _data_lock:
        settings = _get_settings()
        settings.update(updates)
        _write_json(SETTINGS_FILE, settings)
    return jsonify({'ok': True})


# ─── データ エクスポート/インポート API ───
@app.route('/api/export', methods=['GET'])
def api_export():
    with _data_lock:
        data = {
            'version': 1,
            'exportedAt': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'),
            'qaData': _read_json(QA_FILE),
            'tags': _get_tags()
        }
    return jsonify(data)


@app.route('/api/import', methods=['POST'])
def api_import():
    imported = request.json
    if not imported or 'qaData' not in imported:
        return jsonify({'error': 'Invalid format'}), 400
    with _data_lock:
        existing = _read_json(QA_FILE)
        existing_ids = {qa['id'] for qa in existing}
        new_items = [qa for qa in imported['qaData'] if qa['id'] not in existing_ids]
        merged = new_items + existing
        _write_json(QA_FILE, merged)
        if 'tags' in imported:
            tags = _get_tags()
            tag_ids = {t['id'] for t in tags}
            for tag in imported['tags']:
                if tag['id'] not in tag_ids:
                    tags.append(tag)
            _write_json(TAGS_FILE, tags)
    return jsonify({'imported': len(new_items), 'total': len(merged)})


@app.route('/api/clear', methods=['POST'])
def api_clear():
    with _data_lock:
        _write_json(QA_FILE, [])
        _write_json(TAGS_FILE, DEFAULT_TAGS)
    return jsonify({'ok': True})


# ═══════════════════════════════════════════════════
# 壁打ちAI チャット API
# ═══════════════════════════════════════════════════
def _search_relevant_qa(query, max_results=10):
    """Q&Aデータからクエリに関連するエントリを検索"""
    qa_data = _read_json(QA_FILE)
    if not qa_data:
        return []

    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 1]

    scored = []
    for qa in qa_data:
        score = 0
        q_lower = qa.get('question', '').lower()
        a_lower = qa.get('answer', '').lower()
        combined = q_lower + ' ' + a_lower

        # キーワードマッチ
        for kw in keywords:
            if kw in q_lower:
                score += 3
            if kw in a_lower:
                score += 2

        # 完全クエリマッチ
        if query_lower in combined:
            score += 5

        if score > 0:
            scored.append((score, qa))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:max_results]]


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def api_chat():
    # CORS対応（別ドメインの壁打ちAIから呼ばれる）
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        resp.headers['Access-Control-Allow-Methods'] = 'POST'
        return resp

    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', [])

    if not user_message:
        return jsonify({'error': 'メッセージが必要です'}), 400

    settings = _get_settings()
    api_key = settings.get('gemini_api_key')
    if not api_key:
        resp = jsonify({'error': 'Gemini APIキーが設定されていません。統合ツールの設定画面でAPIキーを入力してください。'})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400

    # 関連Q&Aを検索
    relevant_qa = _search_relevant_qa(user_message)
    tags = _get_tags()
    tag_map = {t['id']: t['name'] for t in tags}

    # コンテキスト構築
    context_parts = []
    references = []
    for qa in relevant_qa:
        tag_names = [tag_map.get(tid, '') for tid in qa.get('tags', []) if tag_map.get(tid)]
        tags_str = f" [{', '.join(tag_names)}]" if tag_names else ''
        context_parts.append(f"Q: {qa['question']}\nA: {qa['answer']}{tags_str}")
        references.append({
            'question': qa['question'],
            'answer': qa['answer'][:200] + ('...' if len(qa.get('answer', '')) > 200 else ''),
            'tags': [tag_map.get(tid, '') for tid in qa.get('tags', []) if tag_map.get(tid)]
        })

    context_text = '\n\n---\n\n'.join(context_parts) if context_parts else '（関連するナレッジが見つかりませんでした）'

    system_prompt = f"""あなたは「壁打ちAI」です。ユーザーの仕事やビジネスについて、蓄積されたナレッジベースの知識を使って回答・アドバイスする親しみやすいAIアシスタントです。

## あなたの役割:
- ユーザーのビジネスに関する質問に、ナレッジベースの情報を活用して具体的に回答する
- 壁打ち相手として、アイデアの深掘りや新しい視点を提供する
- 実践的で具体的なアドバイスを心がける
- フランクで親しみやすい口調で話す

## ナレッジベースから取得した関連情報:
{context_text}

## ルール:
1. ナレッジベースの情報がある場合は、それを活用して回答してください
2. ナレッジベースにない情報は、一般的な知識で補完してOKです
3. 回答は簡潔で実用的にまとめてください
4. 必要に応じて箇条書きや見出しを使ってください"""

    # Gemini API呼出し
    contents = []
    for msg in history[-10:]:  # 直近10件の会話履歴
        role = 'user' if msg.get('role') == 'user' else 'model'
        contents.append({'role': role, 'parts': [{'text': msg.get('content', '')}]})
    contents.append({'role': 'user', 'parts': [{'text': user_message}]})

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}'
    payload = json.dumps({
        'systemInstruction': {'parts': [{'text': system_prompt}]},
        'contents': contents,
        'generationConfig': {
            'temperature': 0.7,
            'maxOutputTokens': 4096
        }
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp_api:
            result = json.loads(resp_api.read().decode('utf-8'))
    except Exception as e:
        resp = jsonify({'error': f'Gemini API エラー: {str(e)}'})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500

    ai_response = ''
    try:
        ai_response = result['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        ai_response = 'すみません、回答を生成できませんでした。もう一度試してください。'

    resp = jsonify({
        'response': ai_response,
        'references': references,
        'qa_count': len(relevant_qa)
    })
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


# ═══════════════════════════════════════════════════
# Gemini API でQ&A抽出（サーバーサイド）
# ═══════════════════════════════════════════════════
def _gemini_extract_qa(text, source=''):
    """Gemini API を呼んでテキストからQ&Aペアを抽出"""
    settings = _get_settings()
    api_key = settings.get('gemini_api_key')
    if not api_key:
        print('⚠️ Gemini APIキーが未設定のためQ&A抽出をスキップ')
        return []

    tags = _get_tags()
    tag_names = [t['name'] for t in tags]

    prompt = f"""あなたはプロのナレッジマネジメントの専門家です。以下のテキストから、有用なQ&A（質問と回答）のペアを抽出してください。

ルール:
1. 知識として再利用できる質問と回答のペアを複数抽出してください
2. 質問は具体的な疑問形にしてください
3. 回答はそのまま使えるレベルで具体的にまとめてください
4. 各Q&Aに、利用可能なタグから最も適切なものを1〜3つ付けてください
5. テキストの内容に合うタグがない場合は、新しいタグ名を提案してください

利用可能なタグ: {', '.join(tag_names)}

入力テキスト:
\"\"\"
{text[:15000]}
\"\"\"

出力形式（必ずこのJSON形式で返してください）:
```json
{{
  "qa_pairs": [
    {{
      "question": "質問テキスト",
      "answer": "回答テキスト",
      "tags": ["タグ1", "タグ2"]
    }}
  ]
}}
```

最低3つ、内容が豊富なら10個以上抽出してください。"""

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}'
    payload = json.dumps({
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0.3,
            'maxOutputTokens': 8192,
            'responseMimeType': 'application/json'
        }
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'❌ Gemini API エラー: {e}')
        return []

    content = None
    try:
        content = result['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        print('❌ Gemini APIの応答が空です')
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                print('❌ JSONパースエラー')
                return []
        else:
            print('❌ JSONパースエラー')
            return []

    if 'qa_pairs' not in parsed:
        return []

    # タグ名→IDの変換
    tag_name_to_id = {t['name']: t['id'] for t in _get_tags()}
    qa_items = []
    for pair in parsed['qa_pairs']:
        tag_ids = []
        for tag_name in pair.get('tags', []):
            clean = tag_name.replace('[新規]', '').strip()
            if clean in tag_name_to_id:
                tag_ids.append(tag_name_to_id[clean])
            else:
                new_tag = {'id': f'tag_{uuid.uuid4().hex[:8]}', 'name': clean}
                with _data_lock:
                    tags_list = _get_tags()
                    tags_list.append(new_tag)
                    _write_json(TAGS_FILE, tags_list)
                tag_name_to_id[clean] = new_tag['id']
                tag_ids.append(new_tag['id'])

        qa_items.append({
            'question': pair.get('question', ''),
            'answer': pair.get('answer', ''),
            'tags': tag_ids,
            'source': source
        })

    return qa_items


# ═══════════════════════════════════════════════════
# AI音声用_出力 フォルダ監視
# ═══════════════════════════════════════════════════
def _get_processed_files():
    return set(_read_json(PROCESSED_FILE, []))


def _add_processed_file(filepath):
    with _data_lock:
        processed = _get_processed_files()
        processed.add(filepath)
        _write_json(PROCESSED_FILE, list(processed))


def _watch_ai_output():
    """AI音声用_出力フォルダを定期的にチェックし、新しいテキストをQ&Aに変換"""
    resolved = os.path.realpath(AI_OUTPUT_DIR)
    if not os.path.isdir(resolved):
        print(f'⚠️ AI音声用_出力フォルダが見つかりません: {resolved}')
        print('  → フォルダ監視は無効です')
        return

    print(f'👁️ AI音声用_出力フォルダを監視中: {resolved}')

    while True:
        try:
            processed = _get_processed_files()
            settings = _get_settings()
            api_key = settings.get('gemini_api_key')

            if not api_key:
                time.sleep(60)
                continue

            # 全サブフォルダの .txt ファイルをスキャン
            for category in os.listdir(resolved):
                cat_path = os.path.join(resolved, category)
                if not os.path.isdir(cat_path):
                    continue

                for filename in os.listdir(cat_path):
                    if not filename.endswith('.txt'):
                        continue
                    if filename.endswith('_analysis.txt'):
                        continue  # 分析結果はスキップ

                    filepath = os.path.join(cat_path, filename)
                    if filepath in processed:
                        continue

                    # 新しいファイル発見
                    print(f'📄 新しいテキスト検出: {category}/{filename}')

                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                    except Exception as e:
                        print(f'  ❌ ファイル読み込みエラー: {e}')
                        _add_processed_file(filepath)
                        continue

                    if len(text) < 50:
                        print(f'  ⏭️ テキストが短すぎるためスキップ ({len(text)}文字)')
                        _add_processed_file(filepath)
                        continue

                    source = f'🎤 {category}/{filename.replace(".txt", "")}'
                    print(f'  🤖 Q&A抽出中...')
                    qa_items = _gemini_extract_qa(text, source)

                    if qa_items:
                        now = time.strftime('%Y-%m-%dT%H:%M:%S+09:00')
                        new_items = [{
                            'id': _generate_qa_id(),
                            'question': item['question'],
                            'answer': item['answer'],
                            'tags': item['tags'],
                            'source': item['source'],
                            'createdAt': now,
                            'updatedAt': now
                        } for item in qa_items]

                        with _data_lock:
                            data = _read_json(QA_FILE)
                            data = new_items + data
                            _write_json(QA_FILE, data)

                        print(f'  ✅ {len(new_items)}件のQ&Aを保存しました')
                    else:
                        print(f'  ⚠️ Q&Aを抽出できませんでした')

                    _add_processed_file(filepath)

                    # API レートリミット対策
                    time.sleep(5)

        except Exception as e:
            print(f'❌ フォルダ監視エラー: {e}')

        time.sleep(30)  # 30秒ごとにチェック


# ─── gunicorn対応：アプリ起動時にフォルダ監視を開始 ───
watcher_thread = threading.Thread(target=_watch_ai_output, daemon=True)
watcher_thread.start()

if __name__ == '__main__':
    engine_name = "MLX-Whisper (Apple GPU)" if USE_MLX else "faster-whisper (CPU)"

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   🎙️  文字起こし & ナレッジベース 起動中...     ║")
    print(f"║   ⚡ {engine_name:38s}   ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # 初回のモデルダウンロード
    print("📦 モデルを準備中（初回はダウンロードに少し時間がかかります）...")
    if USE_MLX:
        import mlx_whisper
        import numpy as np
        mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32),
            path_or_hf_repo=MLX_MODELS['base'],
            language='ja',
            verbose=False,
        )
    else:
        get_faster_whisper_model('base')

    port = int(os.environ.get('PORT', 8888))
    print(f"\n✅ 準備完了！ブラウザで http://localhost:{port} へ\n")
    app.run(debug=False, host='0.0.0.0', port=port)
