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
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"
    CLIENTS_DIR: Path = FAISS_INDEX_DIR / "clients"

    # Embedding settings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024

    # ✅ НОВЫЕ настройки для CLIP
    CLIP_MODEL: str = "ViT-B/32"
    VISUAL_EMBEDDING_DIMENSION: int = 512
    DEVICE: str = "auto"  # "cuda", "cpu" или "auto"

    # Chunking settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # FAISS settings
    FAISS_INDEX_TYPE: str = "FlatIP"

    # Document processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: list = [".pdf", ".docx", ".txt", ".md", ".html", ".pptx", ".jpg", ".jpeg", ".png", ".gif",
                                  ".bmp", ".tiff", ".webp"]

    # Download settings
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

    # ✅ НОВЫЕ настройки управления файлами
    KEEP_DOWNLOADED_FILES: bool = False

    # ✅ НОВЫЕ настройки мультимодального поиска
    ENABLE_VISUAL_SEARCH_BY_DEFAULT: bool = False
    MULTIMODAL_SEARCH_TEXT_WEIGHT: float = 0.6
    VISUAL_SIMILARITY_THRESHOLD: float = 0.7
    ENABLE_OCR: bool = True
    OCR_LANGUAGES: list = ['ru', 'en']
    OCR_CONFIDENCE_THRESHOLD: float = 0.5

    def get_client_dir(self, client_id: str) -> Path:
        """Получить основную папку клиента"""
        client_dir = self.CLIENTS_DIR / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        return client_dir

    def get_client_temp_documents_dir(self, client_id: str) -> Path:
        """Получить временную папку документов клиента"""
        if self.KEEP_DOWNLOADED_FILES:
            docs_dir = self.get_client_dir(client_id) / "documents"
        else:
            docs_dir = self.get_client_dir(client_id) / "temp_downloads"

        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_dir

    def get_device_for_processing(self) -> str:
        """Определяет устройство для обработки"""
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE

    class Config:
        env_file = ".env"


# Create directories
def create_directories():
    settings = Settings()
    for path in [settings.DATA_DIR, settings.PROCESSED_DIR,
                 settings.FAISS_INDEX_DIR, settings.CLIENTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


settings = Settings()
create_directories()