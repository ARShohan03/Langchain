from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

st.header("Research Area")
paper=st.selectbox(
    "Select Research Paper Name", ["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis" ]
)
style = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template=load_prompt("template.json")

if st.button("Summerize"):
    chain=template|model
    result=chain.invoke(
        {
            "paper_input" : paper,
            "style_input" : style,
            "length_input" : length
        }
    )
    st.write(result.content)




