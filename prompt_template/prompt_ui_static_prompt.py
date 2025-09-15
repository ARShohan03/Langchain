from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
    # repo_id="google/gemma-2b-it",   # ✅ reliable lightweight chat model
    # task="text-generation",
)
model=ChatHuggingFace(llm=llm)

st.header("Research Area")

user_input=st.text_input("Enter your paper name : ")
if st.button("Summerize"):
    result=model.invoke(user_input)
    st.write(result.content)
