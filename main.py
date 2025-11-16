# SideQuest/LinguaBridge/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel
from langdetect import detect

import os, tempfile
from uuid import uuid4
from gtts import gTTS
from typing import Dict, List, Tuple

from chat_store import save_message, load_chat_history
from feedback import (
    LANGUAGES, STYLES_E2, TTS_LANG_MAP,
    translate_nllb, detect_emotion,
    translate_with_style, cultural_feedback
)

app = FastAPI(title="LinguaBridge Multilingual Chat API")


# =========================
# AUDIO TEMP FOLDER
# =========================
AUDIO_DIR = os.path.join(tempfile.gettempdir(), "linguabridge_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models
# =========================
class TranslationRequest(BaseModel):
    text: str
    src_lang: str | None = None
    tgt_lang: str
    voice_output: bool = False

class RewriteRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    style: str

class CulturalRequest(BaseModel):
    text: str
    src_lang: str


# =========================
# Translation API
# =========================
@app.post("/translate-text")
def translate_text(req: TranslationRequest):
    src = req.src_lang or f"{detect(req.text)}_Latn"
    translated = translate_nllb(req.text, src, req.tgt_lang)
    emotion = detect_emotion(req.text, src)

    audio = None
    if req.voice_output and req.tgt_lang in TTS_LANG_MAP:
        try:
            filename = f"{uuid4().hex}.mp3"
            path = os.path.join(AUDIO_DIR, filename)
            gTTS(translated, lang=TTS_LANG_MAP[req.tgt_lang]).save(path)
            audio = f"/download-audio/{filename}"
        except:
            pass

    return {
        "translated_text": translated,
        "emotion": emotion["label"] if emotion else None,
        "emotion_score": emotion["score"] if emotion else None,
        "audio_file": audio
    }

@app.get("/download-audio/{filename}")
def download_audio(filename: str):
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        return {"error": "Audio missing"}
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


# =========================
# Style Rewrite
# =========================
@app.post("/rewrite-style")
def rewrite(req: RewriteRequest):
    if req.style not in STYLES_E2:
        return {"error": "Invalid style", "allowed": STYLES_E2}

    output = translate_with_style(req.src_lang, req.tgt_lang, req.text, req.style)
    return {"rewritten": output}


# =========================
# Cultural Feedback
# =========================
@app.post("/cultural-feedback")
def cultural_route(req: CulturalRequest):
    return {"feedback": cultural_feedback(req.src_lang, req.text)}


# =========================
# Chat History API
# =========================
@app.get("/chat/history")
def chat_history(chat_id: str):
    msgs = load_chat_history(chat_id)[-20:]
    return [
        {
            "id": m.id,
            "user_id": m.user_id,
            "original_text": m.original_text,
            "src_lang": m.src_lang,
            "created_at": m.created_at.isoformat()
        }
        for m in msgs
    ]


# =========================
# WebSocket Global Chat
# =========================
class ChatClient:
    def __init__(self, websocket: WebSocket, user_id: str, tgt_lang: str):
        self.websocket = websocket
        self.user_id = user_id
        self.tgt_lang = tgt_lang


chat_rooms: Dict[str, List[ChatClient]] = {}


@app.websocket("/ws/chat/{chat_id}")
async def ws_chat(chat_id: str, websocket: WebSocket):
    await websocket.accept()

    try:
        init = await websocket.receive_json()
        user_id = init.get("user_id") or uuid4().hex[:8]
        tgt_lang = init.get("tgt_lang") or "eng_Latn"

        client = ChatClient(websocket, user_id, tgt_lang)
        chat_rooms.setdefault(chat_id, []).append(client)

        while True:
            data = await websocket.receive_json()
            txt = (data.get("text") or "").strip()
            if not txt:
                continue

            src = data.get("src_lang") or f"{detect(txt)}_Latn"

            msg = await run_in_threadpool(save_message, chat_id, user_id, txt, src)

            for other in list(chat_rooms.get(chat_id, [])):
                try:
                    translated = await run_in_threadpool(
                        translate_nllb, txt, src, other.tgt_lang
                    )
                    await other.websocket.send_json({
                        "id": msg.id,
                        "chat_id": chat_id,
                        "from_user": user_id,
                        "original_text": txt,
                        "translated_text": translated,
                        "src_lang": src,
                        "tgt_lang": other.tgt_lang,
                        "created_at": msg.created_at.isoformat()
                    })
                except:
                    pass

    except WebSocketDisconnect:
        if chat_id in chat_rooms:
            chat_rooms[chat_id] = [
                c for c in chat_rooms[chat_id] if c.websocket is not websocket
            ]
            if not chat_rooms[chat_id]:
                del chat_rooms[chat_id]
