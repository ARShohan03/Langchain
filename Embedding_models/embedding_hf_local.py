from langchain_huggingface import HuggingFaceEmbeddings
embedding= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents=[
    "Real madrid is the best club in the world"
]

vector = embedding.embed_documents(documents)

print(str(vector))