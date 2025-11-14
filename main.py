# main.py — FastAPI backend for LinguaBridge: text, audio, and voice translation
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect
from pydub import AudioSegment
import speech_recognition as sr
import torch
import io
import os
import threading
from uuid import uuid4
from gtts import gTTS
import time, glob

app = FastAPI(title="LinguaBridge Multimodal Translator API")

# -------------------------------------------------------------------
# 🌍 Supported Languages — Extended with Indian Low-Resource Languages
# -------------------------------------------------------------------
LANGUAGES = {
    # 🇮🇳 Indian Major + Regional
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",
    "Urdu": "urd_Arab",
    "Assamese": "asm_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "Konkani": "kok_Deva",
    "Maithili": "mai_Deva",
    "Manipuri (Meitei)": "mni_Beng",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi": "snd_Arab",
    "Kashmiri": "kas_Arab",
    "Nepali": "npi_Deva",

    # 🌐 Global languages
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Portuguese": "por_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "ara_Arab"
}

# Text-to-Speech language mapping for gTTS
TTS_LANG_MAP = {
    "eng_Latn": "en", "fra_Latn": "fr", "spa_Latn": "es", "hin_Deva": "hi",
    "zho_Hans": "zh-CN", "deu_Latn": "de", "ara_Arab": "ar", "por_Latn": "pt",
    "jpn_Jpan": "ja", "kor_Hang": "ko", "tam_Taml": "ta", "tel_Telu": "te",
    "kan_Knda": "kn", "mal_Mlym": "ml", "guj_Gujr": "gu", "mar_Deva": "mr",
    "ben_Beng": "bn", "pan_Guru": "pa", "ory_Orya": "or", "urd_Arab": "ur",
}

# -------------------------------------------------------------------
# ⚙️ Model Setup & Thread-Safe Caching
# -------------------------------------------------------------------
MODELS = {
    "small": "facebook/nllb-200-distilled-600M",
    "medium": "facebook/nllb-200-distilled-1.3B",
    "large": "facebook/nllb-200-3.3B"
}

_loaded_models = {}
_model_locks = {}

def get_model(model_size: str, src_lang: str, tgt_lang: str):
    """Thread-safe lazy model loader."""
    model_name = MODELS.get(model_size, MODELS["small"])
    key = f"{model_name}_{src_lang}_{tgt_lang}"

    if key not in _model_locks:
        _model_locks[key] = threading.Lock()

    if key in _loaded_models:
        return _loaded_models[key]

    with _model_locks[key]:
        if key not in _loaded_models:
            print(f"⏳ Loading model for {src_lang} → {tgt_lang} ({model_size}) ...")
            _loaded_models[key] = pipeline(
                "translation",
                model=model_name,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Model ready for {src_lang} → {tgt_lang} ({model_size})!")
    return _loaded_models[key]

# -------------------------------------------------------------------
# 🧹 Background Cleanup Thread for Old Audio Files
# -------------------------------------------------------------------
def clean_old_audio():
    os.makedirs("/tmp/audio", exist_ok=True)
    while True:
        now = time.time()
        for f in glob.glob("/tmp/audio/*.mp3"):
            if os.stat(f).st_mtime < now - 3600:  # delete files older than 1 hour
                os.remove(f)
                print(f"🧹 Deleted old audio file: {f}")
        time.sleep(600)  # every 10 minutes

threading.Thread(target=clean_old_audio, daemon=True).start()

# -------------------------------------------------------------------
# 📦 Request Model
# -------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str
    src_lang: str = None
    tgt_lang: str
    model_size: str = "small"
    voice_output: bool = False  # 🔊 Optional TTS toggle

# -------------------------------------------------------------------
# 🚀 Startup Preload
# -------------------------------------------------------------------
@app.on_event("startup")
async def preload_default_model():
    print("🚀 Preloading English → Hindi model...")
    get_model("small", "eng_Latn", "hin_Deva")
    print("✅ Default model ready!")

# -------------------------------------------------------------------
# 🏠 Root Endpoint
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to LinguaBridge! Use /translate-text, /translate-audio, or /download-audio/{filename}."}

# -------------------------------------------------------------------
# 📝 Text Translation Endpoint (+ Optional Voice)
# -------------------------------------------------------------------
@app.post("/translate-text")
def translate_text(req: TranslationRequest):
    """Translate text with optional voice output."""
    if not req.src_lang:
        detected = detect(req.text)
        req.src_lang = f"{detected}_Latn"
        print(f"🌐 Auto-detected source language: {req.src_lang}")

    if req.src_lang not in LANGUAGES.values() or req.tgt_lang not in LANGUAGES.values():
        return {"error": "Invalid language codes."}

    translator = get_model(req.model_size, req.src_lang, req.tgt_lang)
    result = translator(req.text, max_length=400)
    translated_text = result[0]["translation_text"]

    response = {
        "model": req.model_size,
        "translated_text": translated_text
    }

    # 🔊 Generate TTS output if requested
    if req.voice_output:
        tgt_lang_code = TTS_LANG_MAP.get(req.tgt_lang, "en")
        os.makedirs("/tmp/audio", exist_ok=True)

        filename = f"{uuid4().hex}.mp3"
        file_path = f"/tmp/audio/{filename}"

        try:
            tts = gTTS(translated_text, lang=tgt_lang_code)
            tts.save(file_path)
            print(f"🎧 Audio generated: {file_path}")
            response["audio_file"] = f"/download-audio/{filename}"
        except Exception as e:
            response["voice_error"] = f"TTS failed: {str(e)}"

    return response

# -------------------------------------------------------------------
# 🎧 Audio Translation Endpoint
# -------------------------------------------------------------------
@app.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    target_lang: str = Form("eng_Latn"),
    model_size: str = Form("small")
):
    """Transcribe audio, detect language, and translate."""
    audio_bytes = await file.read()
    file_ext = file.filename.split('.')[-1].lower()

    if file_ext == "mp3":
        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
    else:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_ext)

    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    os.remove(wav_path)

    src_lang_code = detect(text)
    src_lang = f"{src_lang_code}_Latn" if f"{src_lang_code}_Latn" in LANGUAGES.values() else "eng_Latn"

    translator = get_model(model_size, src_lang, target_lang)
    result = translator(text, max_length=400)

    return {
        "source_text": text,
        "source_lang": src_lang,
        "target_lang": target_lang,
        "translated_text": result[0]["translation_text"]
    }

# -------------------------------------------------------------------
# 🎵 Audio Download Endpoint (Dynamic)
# -------------------------------------------------------------------
@app.get("/download-audio/{filename}")
def download_audio(filename: str):
    """Serve dynamically generated TTS files."""
    file_path = f"/tmp/audio/{filename}"
    if not os.path.exists(file_path):
        return {"error": "Audio file not found or expired"}
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
