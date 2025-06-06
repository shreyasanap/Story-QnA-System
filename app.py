import streamlit as st
from config import PDF_PATHS
from core.loader import get_vector_db
from core.agent import build_chain
from core.translator import detect_lang, translate
from core.audio import to_speech
from core.image_gen import generate_image

st.set_page_config(page_title="Chatty", page_icon="🧙")

st.title("Chatty")

query = st.text_input("Ask me a question:")

if st.button("🎤 Speak"):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as mic:
            st.info("Listening…")
            audio = recognizer.listen(mic, timeout=8)
            query = recognizer.recognize_google(audio)
            st.success(f"Heard: {query}")
    except Exception as err:
        st.error(f"Speech error: {err}")

if query:
    src_lang = detect_lang(query)
    en_query = translate(query, target="en") if src_lang != "en" else query

    chain = build_chain(get_vector_db(PDF_PATHS))
    result = chain({"query": en_query})

    if not result.get("source_documents"):
        answer = "Sorry, I Don't Know about that! "
    else:
        answer = result["result"].strip()

    if src_lang != "en":
        answer = translate(answer, target=src_lang)

    st.markdown(f"**Bot:** {answer}")

    audio_file = to_speech(answer, src_lang)
    if audio_file:
        st.audio(audio_file)

    if "fairy tales" not in answer.lower():
        img = generate_image(en_query)
        if img:
            st.image(img, use_column_width=True)
