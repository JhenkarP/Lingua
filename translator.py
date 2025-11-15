# translator.py
import torch
from transformers import pipeline

LANGUAGE_OPTIONS = [
    ("Hindi", "hin_Deva"),
    ("Bengali", "ben_Beng"),
    ("Marathi", "mar_Deva"),
    ("Telugu", "tel_Telu"),
    ("Tamil", "tam_Taml"),
    ("Gujarati", "guj_Gujr"),
    ("Kannada", "kan_Knda"),
    ("Malayalam", "mal_Mlym"),
    ("Punjabi", "pan_Guru"),
    ("Odia", "ory_Orya"),
    ("Urdu", "urd_Arab"),
    ("Assamese", "asm_Beng"),
    ("Bodo", "brx_Deva"),
    ("Dogri", "doi_Deva"),
    ("Konkani", "kok_Deva"),
    ("Maithili", "mai_Deva"),
    ("Manipuri (Meitei)", "mni_Beng"),
    ("Sanskrit", "san_Deva"),
    ("Santali", "sat_Olck"),
    ("Sindhi", "snd_Arab"),
    ("Kashmiri", "kas_Arab"),
    ("Nepali", "npi_Deva"),
    ("English", "eng_Latn"),
    ("French", "fra_Latn"),
    ("Spanish", "spa_Latn"),
    ("German", "deu_Latn"),
    ("Portuguese", "por_Latn"),
    ("Chinese (Simplified)", "zho_Hans"),
    ("Japanese", "jpn_Jpan"),
    ("Korean", "kor_Hang"),
    ("Arabic", "ara_Arab"),
]

def choose_language(prompt):
    print(f"\n{prompt}")
    for i, (lang, code) in enumerate(LANGUAGE_OPTIONS, start=1):
        print(f"{i}. {lang} ({code})")
    choice = int(input("\nEnter the number of your choice: "))
    if 1 <= choice <= len(LANGUAGE_OPTIONS):
        selected_lang = LANGUAGE_OPTIONS[choice - 1]
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
    print("\nTranslated Text:")
    print(result[0]["translation_text"])

if __name__ == "__main__":
    main()
