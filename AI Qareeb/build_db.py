import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# تحميل الملفات
def load_txt_files(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    print(f"✅ Loaded {len(documents)} files")
    return documents
# تحميل البيانات
documents = load_txt_files("./docs")

# تقسيم النصوص
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)

chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc))

print(f"✅ Created {len(chunks)} chunks")

#  تحويل إلى مستندات
processed_chunks = [
    Document(chunk, metadata={"source": "txt"})
    for chunk in chunks
]

print(f"✅ Created {len(processed_chunks)} documents")

# Embedding
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# إنشاء DB
print("🚀 Creating vector store...")

Chroma.from_documents(
    documents=processed_chunks,
    embedding=embedding_model,
    persist_directory="dbv1/chroma_db"
)

print("✅ DB created successfully!")