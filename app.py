import streamlit as st
from main import call_graph  # Import your main code 

# Streamlit page settings
st.set_page_config(page_title="Agentic AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if user_input := st.chat_input("Type your question here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ai_response = call_graph(user_input)
            except Exception as e:
                ai_response = f"⚠️ Error: {e}"
        st.markdown(ai_response)

    # Store AI message
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
