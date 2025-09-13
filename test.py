import json
import speech_recognition as sr
from typing import Dict, Any, Optional, Tuple
from huggingface_hub import InferenceClient
import re
import os
import streamlit as st
from streamlit_chat import message
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import time
from gtts import gTTS

# --- Hugging Face API Token (Loaded from Streamlit Secrets) ---
# Ensure you have set this in your Streamlit Cloud app settings
try:
    HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
    client = InferenceClient(token=HF_API_TOKEN)
except Exception:
    st.error("Hugging Face API Token not found. Please add it to your Streamlit secrets.", icon="🔑")
    st.stop()


# --- Initialize Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Hello! I'm your health assistant. You can ask me about diseases, symptoms, prevention, vaccines, and more!"}
    ]
    st.session_state.audio_enabled = True
    st.session_state.last_audio_hash = None
    st.session_state.last_spoken_index = -1

# --- Load Knowledge Base ---
def load_kb(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Load from any available KB file ---
def load_any_kb() -> Dict[str, Any]:
    """Try multiple filenames and merge if multiple are present."""
    kb: Dict[str, Any] = {}
    candidates = ["dataset.json", "diseases.json"]
    for name in candidates:
        if os.path.exists(name):
            try:
                data = load_kb(name)
                # Merge: later files extend but don't overwrite existing disease entries
                for k, v in data.items():
                    if k not in kb:
                        kb[k] = v
            except Exception:
                continue
    return kb

# Cache the merged knowledge base to avoid reloading on every interaction
@st.cache_data(show_spinner="Loading knowledge base...")
def get_kb_cached() -> Dict[str, Any]:
    return load_any_kb()

# --- Entity normalization and resolution ---
def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())

def resolve_entity_key(kb: Dict[str, Any], entity: str) -> Optional[str]:
    """Resolve user-provided entity to a KB key using aliases and normalization."""
    if not entity:
        return None

    norm_to_key: Dict[str, str] = {}
    for key in kb.keys():
        norm_to_key[_normalize(key)] = key

    # Built-in aliases for common variations
    alias_map = {
        "corona": "covid-19",
        "covid": "covid-19",
        "covid19": "covid-19",
        "coronavirus": "covid-19",
        "sarscov2": "covid-19",
        "tb": "tuberculosis",
        "flu": "influenza",
    }

    norm_entity = _normalize(entity)

    # Direct normalized match
    if norm_entity in norm_to_key:
        return norm_to_key[norm_entity]

    # Alias mapping
    if norm_entity in alias_map:
        alias_norm = _normalize(alias_map[norm_entity])
        if alias_norm in norm_to_key:
            return norm_to_key[alias_norm]

    # Fallback: substring heuristic
    for norm_key, original_key in norm_to_key.items():
        if norm_entity in norm_key or norm_key in norm_entity:
            return original_key

    return None

# --- NLU: Detect Entity and Intent ---
def detect_entity_intent(user_query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (entity, intent)
    entity: disease name from KB
    intent: one of ["symptoms", "prevention", "vaccine", "myth", "outbreak"]
    """
    prompt = f"""
You are an assistant that extracts the main disease and user intent from queries.
Query: "{user_query}"
Output JSON ONLY in the format:
{{ "entity": "disease_name_here", "intent": "symptoms/prevention/vaccine/myth/outbreak" }}
No extra words, no explanation.
"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {"role":"system","content":"You are a helpful NLU assistant."},
                {"role":"user","content":prompt}
            ]
        )
        reply = response.choices[0].message.content.strip()

        # Extract JSON safely
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match:
            data: Dict[str, Any] = json.loads(match.group())
            return data.get("entity"), data.get("intent")
        else:
            return None, None
    except Exception as e:
        print(f"NLU error: {e}")
        return None, None

# --- Get answer from KB ---
def get_answer(kb: Dict[str, Any], entity: str, intent: str) -> Optional[str]:
    # Expand intent synonyms
    intent = (intent or "").strip().lower()
    intent_aliases = {
        "symptom": "symptoms",
        "symptoms": "symptoms",
        "prevention": "prevention",
        "prevent": "prevention",
        "preventions": "prevention",
        "vaccine": "vaccine_schedule",
        "vaccination": "vaccine_schedule",
        "vaccines": "vaccine_schedule",
        "vaccine schedule": "vaccine_schedule",
        "myth": "myths",
        "myths": "myths",
        "outbreak": "outbreak_info",
        "outbreaks": "outbreak_info",
    }
    kb_field = intent_aliases.get(intent)
    if not kb_field:
        return None

    # Resolve entity
    resolved_key = resolve_entity_key(kb, entity)
    if not resolved_key:
        return None

    entry = kb.get(resolved_key, {})
    value = entry.get(kb_field)
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)

# --- Beautify Answer ---
def beautify_answer(answer: str, query: str) -> str:
    prompt = f"""
        You are a friendly healthcare assistant. Answer in English.
        Explain things naturally. For example, if giving symptoms, describe them in full sentences,
        not just a list. Keep tone supportive and positive.
        Answer in English using simple, clear language suitable for rural users. 
        "Explain things naturally in full sentences; do not use bullet points or lists. "
        "Keep tone supportive and positive."
        "Always give a short answer in 2-3 sentences."
        "Do not write long explanations, just keep it brief and supportive."

        User Question: {query}
        Answer Data: {answer}
        Friendly Explanation:
"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {"role":"system","content":"You are a friendly health chatbot."},
                {"role":"user","content":prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Generative error: {e}")
        return f"Here's what I found: {answer}"

# --- NEW: Web-Friendly Text-to-Speech ---
def text_to_audio_autoloader(text: str):
    """Generates audio from text and returns an autoplaying HTML audio tag."""
    if not text or not st.session_state.get('audio_enabled', True):
        return ""
        
    try:
        sound_file = BytesIO()
        tts = gTTS(text, lang='en')
        tts.write_to_fp(sound_file)
        
        sound_file.seek(0)
        audio_base64 = base64.b64encode(sound_file.read()).decode('utf-8')
        
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
        </audio>
        """
        return audio_html
    except Exception as e:
        print(f"gTTS error: {e}")
        return ""

# --- Audio Recording Functions ---
def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text using Google's speech recognition"""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return ""

# --- Streamlit UI Components ---
def display_chat():
    """Display chat messages"""
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- Main Bot Function ---
def process_query(user_query: str):
    """Process user query and generate response"""
    if not user_query.strip():
        return

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Processing your query..."):
        # --- NLU: Extract entity and intent ---
        entity, intent = detect_entity_intent(user_query)
        if not entity or not intent:
            error_msg = "Sorry, I couldn't understand the disease or intent. Could you please rephrase your question?"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return

        # Get answer from knowledge base
        kb = get_kb_cached()
        answer = get_answer(kb, entity, intent)
        if not answer:
            error_msg = f"I don't have information about {intent} for {entity}. Is there something else I can help you with?"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return

        # Generate and display response
        friendly_answer = beautify_answer(answer, user_query)
        st.session_state.messages.append({"role": "assistant", "content": friendly_answer})
        st.rerun()

# --- Post-response actions (MODIFIED) ---
def post_response_actions():
    """Inject an invisible HTML audio player to speak the latest assistant message."""
    try:
        if not st.session_state.get('audio_enabled', True):
            return
        msgs = st.session_state.get('messages', [])
        if not msgs:
            return
        last_idx = len(msgs) - 1
        last_msg = msgs[last_idx]
        
        # Check if it's an assistant message that hasn't been spoken yet
        if last_msg.get("role") == "assistant" and st.session_state.get('last_spoken_index', -1) != last_idx:
            audio_html = text_to_audio_autoloader(last_msg.get("content", ""))
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
                st.session_state.last_spoken_index = last_idx
    except Exception as e:
        st.warning(f"TTS post-render error: {e}")

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="Health Assistant",
        page_icon="🩺",
        layout="wide"
    )

    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.audio_enabled = st.checkbox("Enable Voice", value=True, help="Enable/disable text-to-speech")
        if st.button("Test Voice 🔊"):
            if st.session_state.audio_enabled:
                test_audio_html = text_to_audio_autoloader("Your voice is enabled. I will speak my replies.")
                st.markdown(test_audio_html, unsafe_allow_html=True)
            else:
                st.info("Enable Voice is turned off. Toggle it on to hear replies.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Ask me about:")
        st.markdown("• Symptoms\n• Prevention\n• Vaccines\n• Myths\n• Outbreaks")

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
        if st.session_state.get('last_audio_hash') != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing audio..."):
                transcribed_text = transcribe_audio(audio_bytes)
            
            if transcribed_text:
                time.sleep(0.5)
                process_query(transcribed_text)

if __name__ == "__main__":
    main()
    post_response_actions()