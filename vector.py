from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd

# Load the dataset
df = pd.read_csv("realistic_movie_reviews.csv")

# Load the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# FAISS database location
db_location = "./faiss_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    # Build documents
    documents = []
    for i, row in df.iterrows():
        document = Document(
            page_content=str(row["Title"]) + " " + str(row["Review"]),
            metadata={"rating": row["Rating"], "date": row["Date"]},
        )
        documents.append(document)

    # Create chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Create new FAISS vector store
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    # Persist the index
    vector_store.save_local(db_location)

else:
    # Load existing index from disk
    vector_store = FAISS.load_local(db_location, embeddings,allow_dangerous_deserialization=True)


retriever = vector_store.as_retriever(search_kwargs={"k": 7})
