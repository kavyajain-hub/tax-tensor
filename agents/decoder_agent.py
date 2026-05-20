import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from core.vectorstore import VectorStoreManager

class DecoderAgent:
    def __init__(self):
        # Initialize Llama 3.3 70B via Groq for high-speed, accurate reasoning
        self.llm = ChatGroq(
            temperature=0,  # Zero temperature is critical for legal/tax accuracy
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        # Connect to your local ChromaDB
        self.vsm = VectorStoreManager()
        self.retriever = self.vsm.get_retriever(collection_name="circulars")

    def analyze_impact(self, user_query: str) -> str:
        """Retrieves relevant clauses and generates a structured advisory response."""
        
        # The prompt engineering here forces strict citation and structured markdown
        prompt_template = """
        You are an elite Indian Chartered Accountant and GST compliance expert.
        Analyze the provided government tax notification excerpts to answer the business owner's query.
        
        CRITICAL RULES:
        1. Explain the operational and financial impact in plain English.
        2. Provide actionable advice (e.g., "Update your invoicing software").
        3. ALWAYS cite the exact Section, Clause, or Page Number from the provided text.
        4. If the answer cannot be found in the context, explicitly state: "I cannot find specific guidance for this in the uploaded circular." Do NOT hallucinate tax laws.

        Context:
        {context}

        User Query: {input}

        Format your response exactly like this:
        ### 🤖 Business Impact Analysis
        [Your plain English explanation]
        
        ### ⚠️ Actionable Advice
        [What the business should do to comply]

        > **Verified Citation:**
        > *[Exact reference from text]*
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the LangChain document and retrieval chains
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        
        # Execute the RAG pipeline
        response = retrieval_chain.invoke({"input": user_query})
        return response["answer"]