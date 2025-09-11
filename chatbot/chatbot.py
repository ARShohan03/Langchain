from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

llm_model=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    # task="text-generation",
    temperature=1.5
)
model=ChatHuggingFace(llm=llm_model)

chat_history=[
    SystemMessage(content="you are a helpful assistent. now STARTing the chatting")
]

while True:
    user_input=input("you :")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("ai : ", result.content)

print(chat_history)
    