import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS


load_dotenv()

model = ChatOpenAI(model="gpt-4")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "ResearchPapers", "encoder-decoder.pdf")
persistent_dir = os.path.join(current_dir, "db", "croma_db")


if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist."
        )
    
    # Read the text file 
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Diplay information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating Embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n--- Finishing creating Embeddings ---")

    # Create a vector store using Chroma
    print("\n--- Creating Vector Store ---")
    # vector_store = Chroma.from_documents(
    #     docs, embeddings, persist_directory=persistent_dir
    # )
    vector_store = FAISS.from_documents(
        docs, embeddings
    )
    print("\n--- Finishing creating Vector Store ---")
    print(vector_store.index.ntotal)

else:
    print("Persistent directory already exists. No need to initialize")
















