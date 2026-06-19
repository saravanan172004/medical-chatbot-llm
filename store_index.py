from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load and process PDF data
extracted_data = load_pdf_files("Data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

# Google embeddings (768 dimensions)
embedding = download_embeddings()

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Delete and recreate index with correct dimensions
if pc.has_index(index_name):
    pc.delete_index(index_name)
    print("🗑️ Deleted old index")

pc.create_index(
    name=index_name,
    dimension=384,  # Google text-embedding-004 uses 768 dimensions
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print("✅ Created new index")

index = pc.Index(index_name)

# Clean null metadata to avoid Pinecone errors
for doc in texts_chunk:
    doc.metadata = {
        k: (v if v is not None else "")
        for k, v in doc.metadata.items()
    }

# Upload to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)

print("✅ Data successfully stored in Pinecone!")