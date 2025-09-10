from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents=[
    "Lelouch Lamperouge is a brilliant strategist from Code Geass, known for his Geass power and rebellion against Britannia.",

    "Eren Yeager from Attack on Titan is driven by freedom, transforming into a Titan to fight humanity’s enemies.",

    "Naruto Uzumaki is a ninja of Konoha, famous for his determination, Rasengan, and dream of becoming Hokage.",

    "Goku from Dragon Ball is a Saiyan warrior who protects Earth, known for his Super Saiyan transformations and fighting spirit.",

    "Light Yagami from Death Note discovers a supernatural notebook, using it to enforce his own sense of justice as Kira."
]

query="who is goku?"

doc_vector=embedding.embed_documents(documents)

query_vector=embedding.embed_query(query)

score=cosine_similarity([query_vector], doc_vector)[0]

index, scores=sorted(list(enumerate(score)), key=lambda x : x[1])[-1]

print(query)
print(scores)

print(documents[index])