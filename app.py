import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize session state
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None
if 'past' not in st.session_state:
    st.session_state.past = []  # User messages
if 'generated' not in st.session_state:
    st.session_state.generated = []  # Bot responses

# App UI
st.title("🤖 Smart NLP Chatbot")
st.markdown("Ask me anything. I’ll try to reply intelligently using NLP.")

# Text input + send button
user_input = st.text_input("Your message:", "", key="input")
send = st.button("Send")

if send and user_input:
    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to chat history
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) \
        if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate response
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode bot reply
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Save messages
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

# Display chat messages
with st.expander("📜 Chat History", expanded=True):
    for user_msg, bot_msg in zip(st.session_state.past, st.session_state.generated):
        st.markdown(f"**👤 You:** {user_msg}")
        st.markdown(f"**🤖 Bot:** {bot_msg}")
