import json
import re
import os
import base64
import time
from io import BytesIO
from typing import Dict, Any, Optional, Tuple

import streamlit as st
import speech_recognition as sr
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS

# --- Groq client (key from Streamlit secrets: GROQ_API_KEY) ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Groq API key not found. Add GROQ_API_KEY to your Streamlit secrets.", icon="🔑")
    st.stop()

GROQ_MODEL = "llama-3.1-8b-instant"  # fast; or "llama-3.3-70b-versatile"

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I'm your health assistant. You can ask me about diseases, symptoms, prevention, vaccines, and more!"}
    ]
    st.session_state.audio_enabled = True
    st.session_state.last_audio_hash = None
    st.session_state.last_spoken_index = -1


# --- Knowledge base ---
def load_kb(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_any_kb() -> Dict[str, Any]:
    kb: Dict[str, Any] = {}
    for name in ["dataset.json", "diseases.json"]:
        if os.path.exists(name):
            try:
                for k, v in load_kb(name).items():
                    kb.setdefault(k, v)
            except Exception:
                continue
    return kb


@st.cache_data(show_spinner="Loading knowledge base...")
def get_kb_cached() -> Dict[str, Any]:
    return load_any_kb()


# --- Entity resolution ---
def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


ALIAS_MAP = {
    "covid": "corona", "covid19": "corona", "coronavirus": "corona",
    "sarscov2": "corona", "tb": "tuberculosis", "flu": "influenza",
    "hiv": "hiv aids", "aids": "hiv aids", "hepatitisb": "hepatitis b",
    "alzheimer": "alzheimers", "alzheimer's": "alzheimers",
    "stomachflu": "diarrhea", "cold": "common cold",
}


def resolve_entity_key(kb: Dict[str, Any], entity: str) -> Optional[str]:
    if not entity:
        return None
    norm_to_key = {_normalize(k): k for k in kb.keys()}
    norm_entity = _normalize(entity)

    if norm_entity in norm_to_key:
        return norm_to_key[norm_entity]

    if norm_entity in ALIAS_MAP:
        alias_norm = _normalize(ALIAS_MAP[norm_entity])
        if alias_norm in norm_to_key:
            return norm_to_key[alias_norm]

    # substring fallback (only for entities of 4+ chars to avoid bad matches)
    if len(norm_entity) >= 4:
        for norm_key, original_key in norm_to_key.items():
            if norm_entity in norm_key or norm_key in norm_entity:
                return original_key
    return None


# --- NLU: keyword fallback ---
INTENT_KEYWORDS = {
    "symptoms": ["symptom", "sign", "feel"],
    "prevention": ["prevent", "avoid", "protect"],
    "vaccine": ["vaccin", "immuniz", "shot"],
    "myth": ["myth", "true or false", "is it true"],
    "outbreak": ["outbreak", "epidemic", "spreading"],
}


def keyword_entity_intent(user_query: str, kb: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ql = user_query.lower()
    entity = next((k for k in kb if k in ql), None)
    if not entity:
        words = re.findall(r"[a-z0-9']+", ql)
        for w in words:
            if _normalize(w) in ALIAS_MAP:
                entity = w
                break
    intent = next((i for i, kws in INTENT_KEYWORDS.items()
                   if any(kw in ql for kw in kws)), None)
    return entity, intent


# --- NLU: Groq ---
def detect_entity_intent(user_query: str, kb: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    diseases = ", ".join(kb.keys())
    prompt = f"""Extract the disease and the user's intent from the query.
Query: "{user_query}"
Known diseases: {diseases}
Pick the closest known disease name if possible.
Intent must be one of: symptoms, prevention, vaccine, myth, outbreak.
Return JSON only: {{"entity": "...", "intent": "..."}}"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an NLU assistant. Reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        entity, intent = data.get("entity"), data.get("intent")
        if entity and intent:
            return entity, intent
    except Exception as e:
        print(f"NLU error: {e}")
        st.session_state["last_error"] = f"NLU error: {e}"

    # Fallback if the LLM fails or returns nothing useful
    return keyword_entity_intent(user_query, kb)


# --- Answer lookup ---
INTENT_ALIASES = {
    "symptom": "symptoms", "symptoms": "symptoms",
    "prevention": "prevention", "prevent": "prevention", "preventions": "prevention",
    "vaccine": "vaccine_schedule", "vaccination": "vaccine_schedule",
    "vaccines": "vaccine_schedule", "vaccine schedule": "vaccine_schedule",
    "myth": "myths", "myths": "myths",
    "outbreak": "outbreak_info", "outbreaks": "outbreak_info",
}


def get_answer(kb: Dict[str, Any], entity: str, intent: str) -> Optional[str]:
    kb_field = INTENT_ALIASES.get((intent or "").strip().lower())
    if not kb_field:
        return None
    resolved_key = resolve_entity_key(kb, entity)
    if not resolved_key:
        return None
    value = kb.get(resolved_key, {}).get(kb_field)
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


# --- Friendly rewrite ---
def beautify_answer(answer: str, query: str) -> str:
    prompt = f"""You are a friendly healthcare assistant. Answer in English using simple,
clear language suitable for rural users. Use full sentences, no bullet points or lists.
Keep a supportive, positive tone. Give a short answer in 2-3 sentences.

User Question: {query}
Answer Data: {answer}
Friendly Explanation:"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly health chatbot."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Generative error: {e}")
        return f"Here's what I found: {answer}"


# --- Text-to-speech ---
def text_to_audio_autoloader(text: str):
    if not text or not st.session_state.get("audio_enabled", True):
        return ""
    try:
        sound_file = BytesIO()
        gTTS(text, lang="en").write_to_fp(sound_file)
        sound_file.seek(0)
        audio_base64 = base64.b64encode(sound_file.read()).decode("utf-8")
        return f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
        </audio>
        """
    except Exception as e:
        print(f"gTTS error: {e}")
        return ""


def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data)
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return ""


def display_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# --- Main bot logic ---
def process_query(user_query: str):
    if not user_query.strip():
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Processing your query..."):
        kb = get_kb_cached()
        entity, intent = detect_entity_intent(user_query, kb)

        if not entity or not intent:
            msg = "Sorry, I couldn't understand the disease or intent. Could you please rephrase your question?"
        else:
            answer = get_answer(kb, entity, intent)
            if not answer:
                msg = f"I don't have information about {intent} for {entity}. Is there something else I can help you with?"
            else:
                msg = beautify_answer(answer, user_query)

        st.session_state.messages.append({"role": "assistant", "content": msg})
    st.rerun()


def post_response_actions():
    try:
        if not st.session_state.get("audio_enabled", True):
            return
        msgs = st.session_state.get("messages", [])
        if not msgs:
            return
        last_idx = len(msgs) - 1
        last_msg = msgs[last_idx]
        if last_msg.get("role") == "assistant" and st.session_state.get("last_spoken_index", -1) != last_idx:
            audio_html = text_to_audio_autoloader(last_msg.get("content", ""))
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
                st.session_state.last_spoken_index = last_idx
    except Exception as e:
        st.warning(f"TTS post-render error: {e}")


def main():
    st.set_page_config(page_title="Health Assistant", page_icon="🩺", layout="wide")

    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.audio_enabled = st.checkbox("Enable Voice", value=True)
        if st.button("Test Voice 🔊"):
            if st.session_state.audio_enabled:
                st.markdown(text_to_audio_autoloader("Your voice is enabled."), unsafe_allow_html=True)
            else:
                st.info("Enable Voice is turned off.")
        if st.session_state.get("last_error"):
            st.caption(f"Debug: {st.session_state['last_error']}")
        st.markdown("---")
        st.markdown("### About\nAsk me about:\n\n• Symptoms\n• Prevention\n• Vaccines\n• Myths\n• Outbreaks")

    st.title("🤖 Health Assistant")
    st.caption("Your personal health information assistant")

    display_chat()

    with st.container():
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.chat_input("Type your health question here...")
        with col2:
            audio_bytes = audio_recorder("🎙️", pause_threshold=2.0, key="audio_recorder")

    if user_input:
        process_query(user_input)

    if audio_bytes:
        audio_hash = hash(audio_bytes)
        if st.session_state.get("last_audio_hash") != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing audio..."):
                transcribed_text = transcribe_audio(audio_bytes)
            if transcribed_text:
                time.sleep(0.5)
                process_query(transcribed_text)


if __name__ == "__main__":
    main()
    post_response_actions()
