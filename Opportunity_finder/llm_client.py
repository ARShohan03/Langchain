import requests
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load .env from current and parent directories for robustness
_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")
load_dotenv(_here.parent.parent / ".env")

class MistralClient:
    """A lightweight requests-based client for the modern HuggingFace Inference Router (OpenAI compatible)."""
    
    def __init__(self, token=None):
        self.token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        # UPDATED: Using the standard V1 router endpoint
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.model = "meta-llama/Llama-3.1-8B-Instruct"

    def invoke(self, prompt: str) -> str:
        """Call the HF Router API and return the text response."""
        if not self.token:
            return "[Error] No API token provided. Please add HUGGINGFACEHUB_API_TOKEN to your .env file."
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional scholarship advisor. Respond only with structured data as requested."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=45)
            
            # Robust error handling for varied API responses
            try:
                result = response.json()
            except:
                result = {"error": {"message": response.text}}
            
            if response.status_code != 200:
                if isinstance(result, dict):
                    error_msg = result.get("error", {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(result))
                    else:
                        error_msg = str(error_msg)
                else:
                    error_msg = str(result)
                return f"[API Error] Status {response.status_code}: {error_msg}"
                
            if isinstance(result, dict) and "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            
            return str(result)
        except Exception as e:
            return f"[Connection Error] {str(e)}"

def build_llm_chain(token=None):
    """Factory for the Mistral client."""
    return MistralClient(token)
