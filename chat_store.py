#SideQuest\LinguaBridge\chat_store.py
import os
from uuid import uuid4
from datetime import datetime
from typing import List
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/linguabridge"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, index=True)
    user_id = Column(String, index=True)
    original_text = Column(Text, nullable=False)
    src_lang = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def save_message(chat_id: str, user_id: str, original_text: str, src_lang: str):
    db = SessionLocal()
    try:
        msg = Message(
            id=uuid4().hex,
            chat_id=chat_id,
            user_id=user_id,
            original_text=original_text,
            src_lang=src_lang,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
    finally:
        db.close()

def load_chat_history(chat_id: str) -> List[Message]:
    db = SessionLocal()
    try:
        return (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
            .all()
        )
    finally:
        db.close()
