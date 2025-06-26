import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Mental Health Support Chatbot")
st.markdown("*Empathetic AI companion for mental health support*")

# Sidebar for model path and settings
with st.sidebar:
    st.header("📁 Model Configuration")
    model_path = st.text_input(
        "Model Path/Name:", 
        value="./empathetic-chatbot-model",
        help="Enter the path to your fine-tuned model folder"
    )
    
    # Common model suggestions
    st.markdown("**Common paths:**")
    if st.button("📂 ./my-model"):
        st.session_state.model_path = "./my-model"
    if st.button("📂 ./fine-tuned-model"):
        st.session_state.model_path = "./fine-tuned-model"
    if st.button("🤗 microsoft/DialoGPT-medium"):
        st.session_state.model_path = "microsoft/DialoGPT-medium"
    
    if 'model_path' in st.session_state:
        model_path = st.session_state.model_path
    
    st.markdown("---")
    st.header("⚙️ Generation Settings")
    max_length = st.slider("Response Length", 50, 300, 128)
    temperature = st.slider("Creativity", 0.1, 2.0, 0.7, 0.1)
    do_sample = st.checkbox("Enable Sampling", True)

# Load model with caching
@st.cache_resource
def load_model(model_path):
    try:
        with st.spinner("Loading model... Please wait"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=True  # Only use local files
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load model
if model_path:
    model, tokenizer = load_model(model_path)
else:
    st.warning("Please enter a valid model path in the sidebar.")
    st.stop()

if model is None or tokenizer is None:
    st.error(f"❌ Failed to load the model from: `{model_path}`")
    st.markdown("**Please check:**")
    st.markdown("1. Model folder exists at the specified path")
    st.markdown("2. Folder contains `config.json`, `pytorch_model.bin`, and `tokenizer.json`")
    st.markdown("3. Path is correct (use forward slashes)")
    
    with st.expander("📋 Example folder structure"):
        st.code("""
your-project/
├── app.py
└── empathetic-chatbot-model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
        """)
    st.stop()

# Chat interface
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input("Share your thoughts:", placeholder="How are you feeling today?", key="user_input")

with col2:
    clear_chat = st.button("🗑️ Clear Chat", type="secondary")

if clear_chat:
    st.session_state.chat_history = []
    st.rerun()

# Generate response
if user_input and user_input.strip():
    try:
        with st.spinner("Thinking..."):
            # Prepare input text
            input_text = f"Person: {user_input.strip()}\nBot:"
            
            # Tokenize input
            inputs = tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_response = full_response.split("Bot:")[-1].strip()
            
            # Clean up response
            if not bot_response:
                bot_response = "I understand you're reaching out. Could you tell me more about how you're feeling?"
            
            # Add to chat history
            st.session_state.chat_history.append({
                "user": user_input.strip(),
                "bot": bot_response,
                "timestamp": time.strftime("%H:%M")
            })
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        bot_response = "I apologize, but I'm having trouble responding right now. Please try again."
        st.session_state.chat_history.append({
            "user": user_input.strip(),
            "bot": bot_response,
            "timestamp": time.strftime("%H:%M")
        })

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
        # User message
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.markdown(f"""
                <div style="background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>You ({chat['timestamp']}):</strong><br>
                    {chat['user']}
                </div>
                """, unsafe_allow_html=True)
        
        # Bot response
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🧠 SupportBot ({chat['timestamp']}):</strong><br>
                    {chat['bot']}
                </div>
                """, unsafe_allow_html=True)
        
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    ⚠️ This is an AI chatbot for support purposes only. For serious mental health concerns, please consult a professional.
</div>
""", unsafe_allow_html=True)

# Model info
with st.expander("ℹ️ Model Information"):
    st.write(f"**Model Path:** {model_path}")
    st.write(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.write(f"**PyTorch Version:** {torch.__version__}")
    if torch.cuda.is_available():
        st.write(f"**GPU:** {torch.cuda.get_device_name()}")
    
    # Model details if loaded successfully
    if model is not None:
        try:
            st.write(f"**Model Parameters:** {sum(p.numel() for p in model.parameters()):,}")
            st.write(f"**Vocab Size:** {len(tokenizer)}")
        except:
            pass