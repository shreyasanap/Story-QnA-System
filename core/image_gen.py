import together, base64
from config import TOGETHER_API_KEY, IMAGE_GEN_MODEL

def generate_image(prompt: str):
    """Generate a funny image using the FLUX model on Together."""
    try:
        client = together.Together(api_key=TOGETHER_API_KEY)
        res = client.images.generate(prompt=prompt, model=IMAGE_GEN_MODEL, n=1, size="512x512", steps=8)
        if isinstance(res, dict):
            if "data" in res:
                return res["data"][0]["url"]
            if "output" in res:
                return base64.b64decode(res["output"][0])
        elif hasattr(res, "data") and res.data:
            return res.data[0].url
    except Exception as e:
        print("Image generation error:", e)
    return None
