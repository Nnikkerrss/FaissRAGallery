import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from sentence_transformers import SentenceTransformer
from ..data.chunkers import TextChunk
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSManager:
    """Менеджер для работы с FAISS векторной базой данных"""

    def __init__(self,
                 client_id: str,
                 model_name: str = settings.EMBEDDING_MODEL,
                 index_type: str = settings.FAISS_INDEX_TYPE):
        self.client_id = client_id
        self.model_name = model_name
        self.index_type = index_type
        self.embedding_model = None
        self.index = None
        self.metadata = {}
        self.id_to_chunk_id = {}
        self.chunk_id_to_id = {}
        self.dimension = settings.EMBEDDING_DIMENSION

        # Создаем структуру папок: faiss_index/clients/client_id
        # clients_dir = settings.FAISS_INDEX_DIR / "clients"
        clients_dir = settings.CLIENTS_DIR
        clients_dir.mkdir(parents=True, exist_ok=True)

        self.client_dir = clients_dir / client_id
        self.client_dir.mkdir(parents=True, exist_ok=True)

        # Пути для сохранения с учетом client_id
        self.index_path = self.client_dir / "index.faiss"
        self.metadata_path = self.client_dir / "metadata.pkl"
        self.mappings_path = self.client_dir / "mappings.json"
        self.config_path = self.client_dir / "config.json"

    def initialize_embedding_model(self):
        """Инициализирует модель для создания embeddings"""
        if self.embedding_model is None:
            logger.info(f"Загружаем модель embeddings: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()

    def create_index(self, force_recreate: bool = False):
        """Создает новый FAISS индекс"""
        if self.index is not None and not force_recreate:
            logger.warning("Индекс уже существует. Используйте force_recreate=True для пересоздания")
            return

        logger.info(f"Создаем новый FAISS индекс типа {self.index_type} с размерностью {self.dimension}")

        if self.index_type == "FlatIP":
            # Inner Product (для cosine similarity после нормализации)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "FlatL2":
            # L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World для быстрого поиска
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
        else:
            raise ValueError(f"Неподдерживаемый тип индекса: {self.index_type}")

        # Очищаем метаданные
        self.metadata = {}
        self.id_to_chunk_id = {}
        self.chunk_id_to_id = {}

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Создает embeddings для списка текстов"""
        self.initialize_embedding_model()

        logger.info(f"Создаем embeddings для {len(texts)} текстов")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Нормализуем для cosine similarity (если используем Inner Product)
        if self.index_type == "FlatIP":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: List[TextChunk]) -> List[int]:
        """Добавляет чанки в индекс"""
        if self.index is None:
            self.create_index()

        if not chunks:
            return []

        # Извлекаем тексты для создания embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.create_embeddings(texts)

        # Добавляем в индекс
        start_id = self.index.ntotal
        self.index.add(embeddings)

        # Сохраняем метаданные и маппинги
        added_ids = []
        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            added_ids.append(faiss_id)

            # Сохраняем метаданные
            self.metadata[chunk.chunk_id] = {
                'text': chunk.text,
                'source_file': chunk.source_file,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata,
                'added_date': datetime.now().isoformat(),
                'faiss_id': faiss_id
            }

            # Обновляем маппинги
            self.id_to_chunk_id[faiss_id] = chunk.chunk_id
            self.chunk_id_to_id[chunk.chunk_id] = faiss_id

        logger.info(f"Добавлено {len(chunks)} чанков в индекс. Всего в индексе: {self.index.ntotal}")
        return added_ids

    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Поиск похожих документов"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Индекс пуст или не инициализирован")
            return []

        # Создаем embedding для запроса
        query_embedding = self.create_embeddings([query])

        # Выполняем поиск
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS возвращает -1 для невалидных результатов
                continue

            if score < score_threshold:
                continue

            chunk_id = self.id_to_chunk_id.get(idx)
            if chunk_id and chunk_id in self.metadata:
                result = {
                    'chunk_id': chunk_id,
                    'score': float(score),
                    'text': self.metadata[chunk_id]['text'],
                    'source_file': self.metadata[chunk_id]['source_file'],
                    'metadata': self.metadata[chunk_id]['metadata']
                }
                results.append(result)

        return results

    def remove_chunks(self, chunk_ids: List[str]) -> bool:
        """Удаляет чанки из индекса (пересоздает индекс без удаленных чанков)"""
        if not chunk_ids:
            return True

        # Получаем все чанки кроме удаляемых
        remaining_chunks = []
        for chunk_id, metadata in self.metadata.items():
            if chunk_id not in chunk_ids:
                chunk = TextChunk(
                    text=metadata['text'],
                    chunk_id=chunk_id,
                    source_file=metadata['source_file'],
                    chunk_index=metadata['chunk_index'],
                    metadata=metadata['metadata']
                )
                remaining_chunks.append(chunk)

        # Пересоздаем индекс
        logger.info(f"Пересоздаем индекс для удаления {len(chunk_ids)} чанков")
        self.create_index(force_recreate=True)

        if remaining_chunks:
            self.add_chunks(remaining_chunks)

        return True

    def update_chunk(self, chunk: TextChunk) -> bool:
        """Обновляет существующий чанк"""
        if chunk.chunk_id in self.metadata:
            # Удаляем старый и добавляем новый
            self.remove_chunks([chunk.chunk_id])
            self.add_chunks([chunk])
            return True
        else:
            # Чанк не существует, просто добавляем
            self.add_chunks([chunk])
            return True

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Получает чанк по ID"""
        return self.metadata.get(chunk_id)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Возвращает все чанки"""
        return list(self.metadata.values())

    def get_chunks_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Возвращает все чанки из определенного источника"""
        return [metadata for metadata in self.metadata.values()
                if metadata['source_file'] == source_file]

    def save_index(self):
        """Сохраняет индекс и метаданные на диск"""
        if self.index is None:
            logger.warning("Нет индекса для сохранения")
            return

        logger.info("Сохраняем FAISS индекс и метаданные")

        # Сохраняем индекс
        faiss.write_index(self.index, str(self.index_path))

        # Сохраняем метаданные
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        # Сохраняем маппинги
        mappings_data = {
            'id_to_chunk_id': {str(k): v for k, v in self.id_to_chunk_id.items()},
            'chunk_id_to_id': self.chunk_id_to_id
        }
        with open(self.mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings_data, f, ensure_ascii=False, indent=2)

        # Сохраняем конфигурацию
        config_data = {
            'model_name': self.model_name,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Индекс сохранен: {self.index.ntotal} векторов")

    def load_index(self) -> bool:
        """Загружает индекс и метаданные с диска"""
        if not all(path.exists() for path in [self.index_path, self.metadata_path,
                                              self.mappings_path, self.config_path]):
            logger.warning("Не все файлы индекса найдены")
            return False

        try:
            logger.info("Загружаем FAISS индекс и метаданные")

            # Загружаем конфигурацию
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.model_name = config['model_name']
            self.index_type = config['index_type']
            self.dimension = config['dimension']

            # Загружаем индекс
            self.index = faiss.read_index(str(self.index_path))

            # Загружаем метаданные
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            # Загружаем маппинги
            with open(self.mappings_path, 'r', encoding='utf-8') as f:
                mappings_data = json.load(f)

            self.id_to_chunk_id = {int(k): v for k, v in mappings_data['id_to_chunk_id'].items()}
            self.chunk_id_to_id = mappings_data['chunk_id_to_id']

            logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
            return True

        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Возвращает статистику индекса"""
        if self.index is None:
            return {'status': 'not_initialized'}

        # Группируем по источникам
        sources_stats = {}
        for metadata in self.metadata.values():
            source = metadata['source_file']
            if source not in sources_stats:
                sources_stats[source] = 0
            sources_stats[source] += 1

        return {
            'status': 'ready',
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'model_name': self.model_name,
            'total_chunks': len(self.metadata),
            'sources_count': len(sources_stats),
            'sources_distribution': sources_stats,
            'is_trained': getattr(self.index, 'is_trained', True)
        }

    def clear_index(self):
        """Очищает индекс и все метаданные"""
        logger.info("Очищаем индекс и метаданные")
        self.index = None
        self.metadata = {}
        self.id_to_chunk_id = {}
        self.chunk_id_to_id = {}

        # Удаляем файлы с диска
        for path in [self.index_path, self.metadata_path, self.mappings_path, self.config_path]:
            if path.exists():
                path.unlink()


class EmbeddingManager:
    """Отдельный класс для управления embeddings без привязки к FAISS"""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None

    def initialize_model(self):
        """Инициализирует модель"""
        if self.model is None:
            logger.info(f"Загружаем модель embeddings: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def create_embeddings(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Создает embeddings для текстов"""
        self.initialize_model()

        embeddings = self.model.encode(texts, show_progress_bar=True)

        if normalize:
            # Нормализация для cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.astype(np.float32)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет семантическую близость между двумя текстами"""
        embeddings = self.create_embeddings([text1, text2])
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)