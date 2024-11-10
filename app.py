import streamlit as st
from langchain_ollama import ChatOllama
st.title("Hola Amigo!")
st.write("This is a conversational chatbot")
with st.form("llm-prompt"):
    text = st.text_area("Enter text")
    submit = st.form_submit_button("Submit")

def generate_response(input_text):
    model = ChatOllama(model="llama3.1", base_url="http://localhost:11434/")
    response = model.invoke(input_text)
    return response.content

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []       # initialize the memory for the LLM
if submit and text:
    with st.spinner("Generating response....."):
        response = generate_response(text)
        # we have to append the interaction between the user the LLM agent.
        st.session_state['chat_history'].append({'user': text, "ollama": response})
        st.write(response)