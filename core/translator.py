from deep_translator import GoogleTranslator

def detect_lang(text: str) -> str:
    try:
        return GoogleTranslator().detect(text)
    except Exception:
        return "en"

def translate(text: str, target: str = "en") -> str:
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text
