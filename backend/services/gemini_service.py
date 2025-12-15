import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self):
        """
        Initialize Gemini Service with updated models
        """
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("⚠️ GEMINI_API_KEY not set in .env file")

        # Configure the API with your key
        genai.configure(api_key=api_key)

        # ✅ UPDATED: Use the latest stable Flash model
        # Previous: "gemini-1.5-flash" (retired)
        # Current: "gemini-2.5-flash" (stable)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        # ✅ UPDATED: Use the latest stable text embedding model
        # Previous: "models/embedding-001" (deprecated)
        # Current: "models/text-embedding-005" (stable)
        self.embedding_model = "models/text-embedding-005"

    def generate_embeddings(self, text: str):
        """
        Generate embeddings for documents using the updated embedding model
        
        Args:
            text: The text content to generate embeddings for
            
        Returns:
            List of embedding values
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]

    def generate_query_embedding(self, query: str):
        """
        Generate embeddings for search queries using the updated embedding model
        
        Args:
            query: The search query to generate embeddings for
            
        Returns:
            List of embedding values
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        return result["embedding"]

    def generate_response(self, prompt: str, context: str = ""):
        """
        Generate response using Gemini with context
        
        Args:
            prompt: User's question or prompt
            context: Optional context information for the model
            
        Returns:
            Generated text response
        """
        full_prompt = f"""
You are a helpful assistant for a Physical AI & Humanoid Robotics textbook.

Context:
{context}

User Question:
{prompt}

Answer clearly and accurately.
"""

        response = self.model.generate_content(full_prompt)
        return response.text

    def get_available_models(self):
        """
        Utility method to list available models
        Useful for debugging and verifying API access
        """
        try:
            models = genai.list_models()
            return [model.name for model in models]
        except Exception as e:
            return f"Error fetching models: {str(e)}"