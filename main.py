from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect
import torch
import os
import threading
from uuid import uuid4
from gtts import gTTS
import time, glob
import requests

app = FastAPI(title="LinguaBridge Multimodal Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LANGUAGES = {
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
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Portuguese": "por_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "ara_Arab",
    "Tulu": "tcy_Knda"
}

TTS_LANG_MAP = {
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "deu_Latn": "de",
    "por_Latn": "pt",
    "zho_Hans": "zh-CN",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "ara_Arab": "ar",
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "pan_Guru": "pa",
    "guj_Gujr": "gu",
    "mar_Deva": "mr",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "urd_Arab": "ur",
    "npi_Deva": "ne"
}

MODELS = {
    "small": "facebook/nllb-200-distilled-600M",
    "medium": "facebook/nllb-200-distilled-1.3B"
}

_loaded_models = {}
_model_locks = {}

SPECIAL_LANGS = {
    "brx_Deva",
    "doi_Deva",
    "kok_Deva",
    "sat_Olck",
    "tcy_Knda",
    "ara_Arab"
}

GEMINI_API_KEY = "AIzaSyCPhUfGNeUQVUBrvxr4YDzZD7pfR6kMiAw"


def get_model(model_size: str, src_lang: str, tgt_lang: str):
    model_name = MODELS.get(model_size, MODELS["small"])
    key = f"{model_name}_{src_lang}_{tgt_lang}"
    if key not in _model_locks:
        _model_locks[key] = threading.Lock()
    if key in _loaded_models:
        return _loaded_models[key]
    with _model_locks[key]:
        if key not in _loaded_models:
            _loaded_models[key] = pipeline(
                "translation",
                model=model_name,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1
            )
    return _loaded_models[key]


def clean_old_audio():
    os.makedirs("/tmp/audio", exist_ok=True)
    while True:
        now = time.time()
        for f in glob.glob("/tmp/audio/*.mp3"):
            if os.stat(f).st_mtime < now - 3600:
                os.remove(f)
        time.sleep(600)


threading.Thread(target=clean_old_audio, daemon=True).start()


class TranslationRequest(BaseModel):
    text: str
    src_lang: str = None
    tgt_lang: str
    model_size: str = "small"
    voice_output: bool = False


def gemini_direct_translate(text, src, tgt):
    prompt = (
        "Translate the following text.\n"
        "No creativity. Keep meaning and tone.\n\n"
        f"({src} → {tgt})\n\n"
        f"Text:\n{text}\n\n"
        "Return ONLY the translated text."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload)
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


@app.post("/translate-text")
def translate_text(req: TranslationRequest):
    if not req.src_lang:
        req.src_lang = f"{detect(req.text)}_Latn"

    if req.src_lang in SPECIAL_LANGS or req.tgt_lang in SPECIAL_LANGS:
        translated = gemini_direct_translate(req.text, req.src_lang, req.tgt_lang)
        response = {"translated_text": translated, "via": "gemini"}

        if req.voice_output and req.tgt_lang in TTS_LANG_MAP:
            tgt_lang_code = TTS_LANG_MAP[req.tgt_lang]
            filename = f"{uuid4().hex}.mp3"
            file_path = f"/tmp/audio/{filename}"
            try:
                tts = gTTS(translated, lang=tgt_lang_code)
                tts.save(file_path)
                response["audio_file"] = f"/download-audio/{filename}"
            except:
                pass

        return response

    translator = get_model(req.model_size, req.src_lang, req.tgt_lang)
    result = translator(req.text, max_length=400)
    translated_text = result[0]["translation_text"]

    response = {"translated_text": translated_text, "via": "nllb"}

    if req.voice_output and req.tgt_lang in TTS_LANG_MAP:
        tgt_lang_code = TTS_LANG_MAP[req.tgt_lang]
        filename = f"{uuid4().hex}.mp3"
        file_path = f"/tmp/audio/{filename}"
        try:
            tts = gTTS(translated_text, lang=tgt_lang_code)
            tts.save(file_path)
            response["audio_file"] = f"/download-audio/{filename}"
        except:
            pass

    return response


@app.get("/download-audio/{filename}")
def download_audio(filename: str):
    path = f"/tmp/audio/{filename}"
    if not os.path.exists(path):
        return {"error": "Audio expired or missing."}
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
