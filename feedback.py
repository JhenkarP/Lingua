#SideQuest\LinguaBridge\feedback.py
import torch
from transformers import pipeline
import requests
from langdetect import detect

GEMINI_API_KEY = "AIzaSyCPhUfGNeUQVUBrvxr4YDzZD7pfR6kMiAw"

LANGUAGES = {
    "Hindi": "hin_Deva","Bengali": "ben_Beng","Marathi": "mar_Deva","Telugu": "tel_Telu","Tamil": "tam_Taml",
    "Gujarati": "guj_Gujr","Kannada": "kan_Knda","Malayalam": "mal_Mlym","Punjabi": "pan_Guru","Odia": "ory_Orya",
    "Urdu": "urd_Arab","Assamese": "asm_Beng","Bodo": "brx_Deva","Dogri": "doi_Deva","Konkani": "kok_Deva",
    "Maithili": "mai_Deva","Manipuri (Meitei)": "mni_Beng","Sanskrit": "san_Deva","Santali": "sat_Olck",
    "Sindhi": "snd_Arab","Kashmiri": "kas_Arab","Nepali": "npi_Deva","English": "eng_Latn","French": "fra_Latn",
    "Spanish": "spa_Latn","German": "deu_Latn","Portuguese": "por_Latn","Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan","Korean": "kor_Hang","Arabic": "ara_Arab","Tulu": "tcy_Knda"
}

TTS_LANG_MAP = {
    "eng_Latn": "en","fra_Latn": "fr","spa_Latn": "es","deu_Latn": "de","por_Latn": "pt","zho_Hans": "zh-CN",
    "jpn_Jpan": "ja","kor_Hang": "ko","ara_Arab": "ar","hin_Deva": "hi","ben_Beng": "bn","pan_Guru": "pa",
    "guj_Gujr": "gu","mar_Deva": "mr","tam_Taml": "ta","tel_Telu": "te","kan_Knda": "kn","mal_Mlym": "ml",
    "urd_Arab": "ur","npi_Deva": "ne"
}

STYLES_E2 = [
    "joy","sadness","anger","fear","disgust",
    "sarcastic","polite","formal","casual","motivational"
]

_TRANSLATOR_CACHE = {}

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def get_translator(src_lang: str, tgt_lang: str):
    key = (src_lang, tgt_lang)
    if key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[key]
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dtype=torch.float16,
        device=0 if torch.cuda.is_available() else -1
    )
    _TRANSLATOR_CACHE[key] = translator
    return translator

def translate_nllb(text: str, src_lang: str, tgt_lang: str):
    try:
        translator = get_translator(src_lang, tgt_lang)
        return translator(text, max_length=400)[0]["translation_text"]
    except Exception:
        return text

def gemini_send_prompt(prompt: str):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def detect_emotion(text: str, src_lang: str):
    try:
        if src_lang == "eng_Latn":
            emo_input = text
        else:
            emo_input = translate_nllb(text, src_lang, "eng_Latn")
        r = emotion_model(emo_input)[0]
        return {"label": r["label"].lower(), "score": float(r["score"])}
    except Exception:
        return None

def translate_with_style(src_lang: str, tgt_lang: str, text: str, style: str) -> str:
    literal = translate_nllb(text, src_lang, tgt_lang)
    prompt = (
        f"Rewrite the text in {tgt_lang}. Tone: {style}.\n\n"
        f"Original:\n{text}\n\n"
        f"Literal:\n{literal}\n\n"
        "Return only rewritten output."
    )
    return gemini_send_prompt(prompt).strip()

def cultural_feedback(src_lang: str, text: str):
    name = None
    for n, code in LANGUAGES.items():
        if code == src_lang:
            name = n
            break
    if not name:
        name = src_lang
    prompt = (
        f"Linguistic and cultural insight for a sentence in {name} ({src_lang}):\n"
        f"\"{text}\"\n\nShort, 3-4 sentences, include a fun fact."
    )
    return gemini_send_prompt(prompt).strip()
