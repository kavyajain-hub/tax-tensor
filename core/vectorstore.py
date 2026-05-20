import os
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class VectorStoreManager:
    def __init__(self):
        # Local persistent storage
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        
        # Configure explicit local caching for HuggingFace embeddings
        # This prevents redundant downloads and makes Docker containerization easier
        self.cache_dir = os.path.join(os.getcwd(), "data", "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder=self.cache_dir
        )
        
    def extract_pages_from_pdf(self, pdf_path: str) -> list:
        """
        Extracts text while maintaining page boundaries for accurate metadata tagging.
        Returns a list of dictionaries: [{'text': '...', 'page': 1}, ...]
        """
        pages_data = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(layout=True)
                    if text:
                        # We still inject the visual page marker for the LLM's context window,
                        # but we also return the integer for the vector database metadata.
                        text_with_marker = f"\n--- [PAGE {page_num}] ---\n{text}"
                        pages_data.append({"text": text_with_marker, "page": page_num})
            return pages_data
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def index_document(self, file_path: str, collection_name: str = "circulars"):
        """
        Chunks the extracted legal text, checks for duplicates, and indexes it.
        """
        filename = os.path.basename(file_path)
        collection_path = os.path.join(self.persist_directory, collection_name)
        
        # --- RE-INDEXING GUARD ---
        # Check if the database exists and already contains this file
        if os.path.exists(collection_path):
            try:
                existing_db = Chroma(
                    persist_directory=collection_path,
                    embedding_function=self.embeddings
                )
                # Query Chroma for any documents matching this source filename
                existing_docs = existing_db.get(where={"source": filename})
                
                if existing_docs and len(existing_docs['ids']) > 0:
                    print(f"[{filename}] already indexed. Skipping to prevent duplication.")
                    return existing_db
            except Exception as e:
                print(f"Warning during duplication check: {str(e)}. Proceeding with index.")

        # Extract and split
        pages_data = self.extract_pages_from_pdf(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # --- CONSISTENT METADATA INJECTION ---
        docs = []
        for page in pages_data:
            chunks = text_splitter.create_documents(
                texts=[page["text"]], 
                metadatas=[{"source": filename, "page": page["page"]}]
            )
            docs.extend(chunks)
        
        # --- ERROR HANDLING FOR PERSISTENCE ---
        try:
            vector_db = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=collection_path
            )
            return vector_db
        except Exception as e:
            # Catching SQLite locks, permission errors, or corrupted DB states
            raise Exception(f"Database persistence failed: {str(e)}")
        
    def get_retriever(self, collection_name: str = "circulars"):
        """
        Returns a configured retriever object for LangChain RAG pipelines.
        """
        vector_db = Chroma(
            persist_directory=os.path.join(self.persist_directory, collection_name),
            embedding_function=self.embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": 5})