#mp3.py
import speech_recognition as sr
from langdetect import detect
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment
import os

def select_file():
    Tk().withdraw()  
    filename = askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    return filename

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.rsplit('.', 1)[0] + ".wav"
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    # Convert if mp3
    if file_path.endswith(".mp3"):
        print("Converting MP3 to WAV...")
        file_path = convert_mp3_to_wav(file_path)

    print("Transcribing...")
    transcription = transcribe_audio(file_path)
    if transcription is None:
        print("Could not transcribe the audio.")
        return

    print(f"Transcription: {transcription}")

    print("Detecting language...")
    language = detect(transcription)
    print(f"Detected Language Code: {language}")

    # Cleanup converted wav if needed
    if file_path.endswith(".wav") and "converted" in file_path:
        os.remove(file_path)

if __name__ == "__main__":
    main()