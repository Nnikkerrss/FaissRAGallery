import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"
    CLIENTS_DIR: Path = FAISS_INDEX_DIR / "clients"  # Новая папка для клиентов

    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # FAISS settings
    FAISS_INDEX_TYPE: str = "FlatIP"  # Inner Product для cosine similarity

    # Document processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: list = [".pdf", ".docx", ".txt", ".md", ".html", ".jpg", ".jpeg", ".png", ".gif", ".bmp"]

    # Download settings
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

    class Config:
        env_file = ".env"


# Create directories
def create_directories():
    settings = Settings()
    for path in [settings.DATA_DIR, settings.DOCUMENTS_DIR,
                 settings.PROCESSED_DIR, settings.FAISS_INDEX_DIR, settings.CLIENTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


settings = Settings()
create_directories()