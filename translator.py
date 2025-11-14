#translator.py
import torch
from transformers import pipeline

LANGUAGES = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Hindi": "hin_Deva",
    "Chinese (Simplified)": "zho_Hans",
    "German": "deu_Latn",
    "Arabic": "ara_Arab",
    "Portuguese": "por_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
}

def choose_language(prompt):
    print(f"\n{prompt}")
    for i, (lang, code) in enumerate(LANGUAGES.items(), start=1):
        print(f"{i}. {lang} ({code})")

    choice = int(input("\nEnter the number of your choice: "))
    if 1 <= choice <= len(LANGUAGES):
        selected_lang = list(LANGUAGES.items())[choice - 1]
        print(f"Selected: {selected_lang[0]}\n")
        return selected_lang[1]
    else:
        print("Invalid choice. Please restart the program.")
        exit()

def main():
    print("=== NLLB-200 CLI Translator ===")

    src_lang = choose_language("Select the source language:")
    tgt_lang = choose_language("Select the target language:")
    text = input("Enter the text to translate:\n> ")

    print("\nTranslating... Please wait.\n")

    translator = pipeline(
        task="translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dtype=torch.float16,
        device=0 if torch.cuda.is_available() else -1
    )

    result = translator(text, max_length=400)
    print("Translated Text:")
    print(result[0]["translation_text"])

if __name__ == "__main__":
    main()