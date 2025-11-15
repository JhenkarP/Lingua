# text_translator_api.py
import torch
from transformers import pipeline
import requests

# Available languages (expand as needed)
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

# Available NLLB models on Hugging Face
MODELS = {
    "NLLB-200 Distilled 600M (fast, small)": "facebook/nllb-200-distilled-600M",
    "NLLB-200 Distilled 1.3B (balanced)": "facebook/nllb-200-distilled-1.3B",
    "NLLB-200 3.3B (high quality, large)": "facebook/nllb-200-3.3B"
}

def choose_option(prompt, options):
    print(f"\n{prompt}")
    for i, name in enumerate(options.keys(), start=1):
        print(f"{i}. {name}")

    choice = int(input("\nEnter the number of your choice: "))
    if 1 <= choice <= len(options):
        selected = list(options.items())[choice - 1]
        print(f"Selected: {selected[0]}\n")
        return selected[1]
    else:
        print("Invalid choice. Please restart the program.")
        exit()

def translate_via_api(model_choice, hf_token, text, src_lang, tgt_lang):
    API_URL = f"https://api-inference.huggingface.co/models/{model_choice}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        },
        "options": {
            "wait_for_model": True   # IMPORTANT for NLLB models
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()

    if isinstance(result, list) and "translation_text" in result[0]:
        return result[0]["translation_text"]
    elif "error" in result:
        raise RuntimeError(f"API Error: {result['error']}")
    else:
        raise RuntimeError("Unexpected API response format")

def main():
    print("=== NLLB CLI Translator ===")

    # Choose run mode
    run_mode = choose_option("Select run mode:", {
        "Local": "local",
        "Hugging Face API": "api"
    })

    # Choose model and languages
    model_choice = choose_option("Select the NLLB model to use:", MODELS)
    src_lang = choose_option("Select the source language:", LANGUAGES)
    tgt_lang = choose_option("Select the target language:", LANGUAGES)

    text = input("Enter the text to translate:\n> ")

    # Local mode
    if run_mode == "local":
        print("\nLoading model locally... This may take a moment.\n")
        translator = pipeline(
            task="translation",
            model=model_choice,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Translating...\n")
        result = translator(text, max_length=400)
        translated_text = result[0]["translation_text"]

    else:  # Hugging Face API mode
        hf_token = input("Enter your Hugging Face API token:\n> ").strip()
        print("\nCalling Hugging Face Inference API...\n")
        try:
            translated_text = translate_via_api(
                model_choice, hf_token, text, src_lang, tgt_lang
            )
        except Exception as e:
            print(f"Translation API call failed: {e}")
            return

    print("Translated Text:\n" + translated_text)

if __name__ == "__main__":
    main()
