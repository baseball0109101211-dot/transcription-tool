import os
import sys
import uuid
import tempfile
import threading
import time
import platform
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


if __name__ == '__main__':
    engine_name = "MLX-Whisper (Apple GPU)" if USE_MLX else "faster-whisper (CPU)"

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   🎙️  文字起こしツール 起動中...               ║")
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
