import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import speech_recognition as sr
import os


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None
if 'past' not in st.session_state:
    st.session_state.past = []
if 'generated' not in st.session_state:
    st.session_state.generated = []


st.title("🤖 Smart NLP Chatbot")
st.markdown("Chat with emojis 😄, voice 🎤, and save your session 💾.")


def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Sorry, I couldn't understand."


st.sidebar.subheader("⚙️ Settings")
theme = st.sidebar.radio("Select Theme (refresh app):", ["Light", "Dark"])
st.sidebar.markdown("To change theme instantly, edit `.streamlit/config.toml`")


if st.sidebar.button("🔁 Reset Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.past = []
    st.session_state.generated = []
    st.experimental_rerun()


if st.sidebar.button("💾 Save Chat"):
    with open("chat_history.txt", "w", encoding='utf-8') as f:
        for user_msg, bot_msg in zip(st.session_state.past, st.session_state.generated):
            f.write(f"You: {user_msg}\nBot: {bot_msg}\n\n")
    st.sidebar.success("Chat saved to chat_history.txt")


col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("💬 Your message:", key="input")
with col2:
    if st.button("🎤 Voice"):
        user_input = get_voice_input()
        st.session_state.input = user_input


if st.button("Send") and user_input:
    # Tokenize
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )


    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)


with st.expander("📜 Chat History", expanded=True):
    for user_msg, bot_msg in zip(st.session_state.past, st.session_state.generated):
        st.markdown(f"**👤 You:** {user_msg}")
        st.markdown(f"**🤖 Bot:** {bot_msg} 😊")

