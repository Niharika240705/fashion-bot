"""
Streamlit Fashion Bot Application
Features:
- Multiple model selection (TF-IDF, Word2Vec, BERT, API)
- Advanced NLP analysis display
- Real-time chat interface
"""

import streamlit as st
from chatbot_advanced import initialize_bot, get_bot_response
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Luxury Fashion Bot",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = None
    st.session_state.current_model = "word2vec"
    st.session_state.chat_history = []

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model selection
    model_choice = st.radio(
        "Select Model:",
        ["word2vec", "tfidf", "bert"],
        index=0
    )
    
    if st.button("Load Model", use_container_width=True):
        with st.spinner(f"Loading {model_choice} model..."):
            try:
                st.session_state.bot = initialize_bot(model_choice)
                st.session_state.current_model = model_choice
                st.success(f"✓ {model_choice.upper()} model loaded!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    st.divider()
    
    # Chat settings
    st.subheader("Chat Settings")
    use_advanced_nlp = st.checkbox("Advanced NLP Analysis", value=True)
    use_internet = st.checkbox("Internet Search", value=False)
    
    # Clear history button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.title("👗 Luxury Fashion Bot")
st.markdown(f"**Current Model:** `{st.session_state.current_model.upper()}`")

# Initialize bot if needed
if st.session_state.bot is None:
    with st.spinner("Initializing bot with default model..."):
        try:
            st.session_state.bot = initialize_bot("word2vec")
            st.session_state.current_model = "word2vec"
            st.success("✓ Bot initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing bot: {str(e)}")
            st.warning("Please check that training has been completed.")

# Chat interface
if st.session_state.bot is not None:
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Bot:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
                
                if "analysis" in message and message["analysis"]:
                    with st.expander("📊 NLP Analysis"):
                        analysis = message["analysis"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if "entities" in analysis:
                                st.write("**Named Entities:**")
                                st.write(analysis["entities"])
                            if "keywords" in analysis:
                                st.write("**Top Keywords:**")
                                st.write(analysis["keywords"][:5])
                        
                        with col2:
                            if "intent" in analysis:
                                st.write("**Detected Intent:**")
                                st.write(f"`{analysis['intent']}`")
                            if "sentiment" in analysis:
                                st.write("**Sentiment:**")
                                st.write(f"`{analysis['sentiment']}`")
    
    st.divider()
    
    # Input section
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message:",
            placeholder="Ask about fashion, trends, or get recommendations...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get bot response
        with st.spinner("Thinking..."):
            try:
                response_data = get_bot_response(
                    user_input,
                    st.session_state.bot,
                    use_advanced_nlp=use_advanced_nlp,
                    use_internet=use_internet
                )
                
                bot_message = {
                    "role": "bot",
                    "content": response_data.get("response", "I couldn't generate a response."),
                }
                
                if use_advanced_nlp and "analysis" in response_data:
                    bot_message["analysis"] = response_data["analysis"]
                
                st.session_state.chat_history.append(bot_message)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.warning("⚠️ Bot not initialized. Please load a model using the sidebar.")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Luxury Fashion Model-based Bot")
with col2:
    st.caption(f"Messages: {len([m for m in st.session_state.chat_history if m['role'] == 'user'])}")
with col3:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
