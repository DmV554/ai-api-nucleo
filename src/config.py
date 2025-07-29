
import os
from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    """
    Loads and validates application settings from environment variables
    or a .env file.
    """
    # --- Provider Configuration ---
    # Use 'ollama' for local models, 'openai' for OpenAI API
    LLM_PROVIDER: str = "ollama"
    # Use 'local' for SentenceTransformers, 'openai' for OpenAI API
    EMBEDDING_PROVIDER: str = "local"

    # --- OpenAI Configuration ---
    OPENAI_API_KEY: SecretStr = "sk-..."
    OPENAI_LLM_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # --- Local/Ollama Model Configuration ---
    OLLAMA_LLM_MODEL: str = "llama3"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SPARSE_EMBEDDING_MODEL: str = "naver/splade-v2-distil"
    RANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_DIM: int = 384 # For all-MiniLM-L6-v2

    # --- Vector Store Configuration ---
    # It's important that this path is absolute for consistency.
    VECTOR_STORE_PATH: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../vector_stores')
    )

    # --- API Configuration ---
    API_V1_STR: str = "/api/v1"

    class Config:
        # This tells Pydantic to load variables from a .env file.
        env_file = ".env"
        # env_file_encoding = 'utf-8' # Uncomment if you have encoding issues
        case_sensitive = True

# Create a single, importable instance of the settings
settings = Settings()
