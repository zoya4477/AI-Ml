import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Page configuration
st.set_page_config(page_title="Mental Health Support Bot", page_icon="💙", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f0f8ff;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input {
        background-color: #f9f9f9;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #d3d3d3;
    }
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 style='text-align: center; color: #004080;'>💙 Mental Health Support Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This chatbot provides a safe, caring space to talk about how you feel. 💬</p>", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "/workspaces/AI-Ml/empathetic-chatbot-model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

if model is None:
    st.stop()

# User input box
st.markdown("### How are you feeling today?")
user_input = st.text_input("You:", placeholder="Type here...", key="user_input")

# Detect sensitive emotional terms
sensitive_keywords = ["depressed", "anxious", "suicidal", "hopeless", "worthless", "tired", "lonely", "lost", "scared", "broken"]

if user_input:
    # Handle sensitive input
    if any(word in user_input.lower() for word in sensitive_keywords):
        st.markdown("<div style='background-color:#ffe6e6;padding:15px;border-radius:10px;'>"
                    "<strong>Bot:</strong> I'm really sorry you're feeling this way. 💙 You're not alone. "
                    "Please consider talking to someone you trust or reaching out to a mental health professional. You matter."
                    "</div>", unsafe_allow_html=True)
    else:
        # Empathetic chatbot prompt
        prompt = f"""You are a kind and empathetic mental health support assistant. You always listen carefully and respond with warmth, encouragement, and understanding.

Human: {user_input}
Bot:"""

        inputs = tokenizer.encode(prompt, return_tensors="pt")

        with st.spinner("Thinking..."):
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=150,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_reply = response.split("Bot:")[-1].strip()

        st.markdown(f"<div style='background-color:#e6f2ff;padding:15px;border-radius:10px;'>"
                    f"<strong>Bot:</strong> {bot_reply}</div>", unsafe_allow_html=True)

#  Footer
st.markdown("<br><hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>This is not a substitute for professional help. If you are in crisis, please seek immediate assistance. 💙</p>", unsafe_allow_html=True)
