# app.py
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("🧠 Mental Health Support Chatbot")

model = AutoModelForCausalLM.from_pretrained("empathetic-chatbot-model")
tokenizer = AutoTokenizer.from_pretrained("empathetic-chatbot-model")

user_input = st.text_input("You:", "")

if user_input:
    input_text = f"Person: {user_input}\nYou:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=128, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).split("You:")[-1]
    st.text_area("SupportBot:", reply.strip(), height=100)
