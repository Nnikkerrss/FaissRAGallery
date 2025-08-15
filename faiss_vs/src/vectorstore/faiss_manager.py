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
    """Менеджер для работы с FAISS векторной базой данных с поддержкой мультимодальности"""

    def __init__(self,
                 client_id: str,
                 model_name: str = settings.EMBEDDING_MODEL,
                 index_type: str = settings.FAISS_INDEX_TYPE,
                 enable_visual_search: bool = False):  # ✅ ДОБАВЛЕН параметр

        self.client_id = client_id
        self.model_name = model_name
        self.index_type = index_type
        self.enable_visual_search = enable_visual_search  # ✅ НОВОЕ

        # Модели и индексы
        self.embedding_model = None
        self.index = None  # Старый единый индекс (для совместимости)
        self.text_index = None  # ✅ НОВОЕ: Для текстовых векторов
        self.visual_index = None  # ✅ НОВОЕ: Для визуальных векторов

        # Метаданные и маппинги
        self.metadata = {}
        self.id_to_chunk_id = {}
        self.chunk_id_to_id = {}
        # ✅ НОВЫЕ маппинги для мультимодального режима
        self.text_id_to_chunk_id = {}
        self.visual_id_to_chunk_id = {}
        self.chunk_id_to_ids = {}  # {chunk_id: {'text_id': X, 'visual_id': Y}}

        # Размерности
        self.dimension = settings.EMBEDDING_DIMENSION
        self.text_dimension = settings.EMBEDDING_DIMENSION
        self.visual_dimension = settings.VISUAL_EMBEDDING_DIMENSION  # ✅ НОВОЕ

        # Создаем структуру папок: faiss_index/clients/client_id
        self.client_dir = settings.CLIENTS_DIR / client_id
        self.client_dir.mkdir(parents=True, exist_ok=True)

        # Пути для сохранения
        self.index_path = self.client_dir / "index.faiss"  # Старый путь
        self.text_index_path = self.client_dir / "text_index.faiss"  # ✅ НОВОЕ
        self.visual_index_path = self.client_dir / "visual_index.faiss"  # ✅ НОВОЕ
        self.metadata_path = self.client_dir / "metadata.pkl"
        self.mappings_path = self.client_dir / "mappings.json"
        self.config_path = self.client_dir / "config.json"

    def initialize_embedding_model(self):
        """Инициализирует модель для создания embeddings"""
        if self.embedding_model is None:
            logger.info(f"Загружаем модель embeddings: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.text_dimension = self.embedding_model.get_sentence_embedding_dimension()

    def create_index(self, force_recreate: bool = False):
        """Создает новый FAISS индекс"""
        if not force_recreate:
            if not self.enable_visual_search and self.index is not None:
                logger.warning("Индекс уже существует. Используйте force_recreate=True для пересоздания")
                return
            elif self.enable_visual_search and (self.text_index is not None or self.visual_index is not None):
                logger.warning("Мультимодальные индексы уже существуют. Используйте force_recreate=True")
                return

        logger.info(f"Создаем FAISS индекс(ы) типа {self.index_type}")

        if self.enable_visual_search:
            # ✅ МУЛЬТИМОДАЛЬНЫЙ режим: создаем два индекса
            logger.info("Создаем мультимодальные индексы")

            if self.index_type == "FlatIP":
                self.text_index = faiss.IndexFlatIP(self.text_dimension)
                self.visual_index = faiss.IndexFlatIP(self.visual_dimension)
            elif self.index_type == "FlatL2":
                self.text_index = faiss.IndexFlatL2(self.text_dimension)
                self.visual_index = faiss.IndexFlatL2(self.visual_dimension)
            elif self.index_type == "HNSW":
                self.text_index = faiss.IndexHNSWFlat(self.text_dimension, 32)
                self.visual_index = faiss.IndexHNSWFlat(self.visual_dimension, 32)
            else:
                raise ValueError(f"Неподдерживаемый тип индекса: {self.index_type}")

            logger.info(f"✅ Создан текстовый индекс: {self.text_dimension}D")
            logger.info(f"✅ Создан визуальный индекс: {self.visual_dimension}D")

            # Очищаем мультимодальные метаданные
            self.text_id_to_chunk_id = {}
            self.visual_id_to_chunk_id = {}
            self.chunk_id_to_ids = {}

        else:
            # ОБЫЧНЫЙ режим: создаем единый индекс (как раньше)
            if self.index_type == "FlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "FlatL2":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
            else:
                raise ValueError(f"Неподдерживаемый тип индекса: {self.index_type}")

            logger.info(f"✅ Создан единый индекс: {self.dimension}D")

            # Очищаем старые метаданные
            self.id_to_chunk_id = {}
            self.chunk_id_to_id = {}

        # Очищаем общие метаданные
        self.metadata = {}

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

        # 🔧 ИСПРАВЛЯЕМ: Создаем тексты для векторизации ВКЛЮЧАЯ МЕТАДАННЫЕ
        texts_for_embedding = []

        for chunk in chunks:
            # Собираем полный текст для векторизации
            full_text_parts = [chunk.text]  # Основной текст чанка

            # Добавляем важные метаданные к тексту для векторизации
            metadata = chunk.metadata

            if metadata.get('title'):
                full_text_parts.append(f"Заголовок: {metadata['title']}")

            if metadata.get('description'):
                full_text_parts.append(f"Описание: {metadata['description']}")

            if metadata.get('category'):
                full_text_parts.append(f"Категория: {metadata['category']}")

            if metadata.get('parent'):
                full_text_parts.append(f"Раздел: {metadata['parent']}")

            # Объединяем все части
            combined_text = " ".join(full_text_parts)
            texts_for_embedding.append(combined_text)

            # 🔧 ПРИНУДИТЕЛЬНЫЙ ВЫВОД в stdout И в logger
            debug_message = f"🔧 ВЕКТОРИЗАЦИЯ {chunk.source_file} (чанк {chunk.chunk_index}):"
            # print(debug_message)  # Прямой вывод в консоль
            logger.info(debug_message)  # Через logger
            #
            info_message = f"   📝 Исходный: {len(chunk.text)} символов | Итоговый: {len(combined_text)} символов"
            # print(info_message)
            logger.info(info_message)
            #
            meta_message = f"   🏷️ title='{metadata.get('title', 'НЕТ')}' | description='{metadata.get('description', 'НЕТ')}' | category='{metadata.get('category', 'НЕТ')}'"
            # print(meta_message)
            logger.info(meta_message)

            if len(combined_text) > len(chunk.text) + 50:  # Если метаданные добавились
                added_meta = combined_text[len(chunk.text):100] + "..."
                added_message = f"   ➕ Добавлено к тексту: '{added_meta}'"
                # print(added_message)
                logger.info(added_message)
            else:
                print("   ⚠️ МЕТАДАННЫЕ НЕ ДОБАВИЛИСЬ К ТЕКСТУ!")
                logger.warning("   ⚠️ МЕТАДАННЫЕ НЕ ДОБАВИЛИСЬ К ТЕКСТУ!")

        # Создаем embeddings из расширенных текстов
        # print(f"\n🔄 Создаем embeddings для {len(texts_for_embedding)} расширенных текстов...")
        logger.info(f"Создаем embeddings для {len(texts_for_embedding)} расширенных текстов")

        embeddings = self.create_embeddings(texts_for_embedding)

        # Добавляем в индекс
        start_id = self.index.ntotal
        self.index.add(embeddings)

        # print(f"✅ Embeddings созданы и добавлены в FAISS индекс")
        logger.info(f"Embeddings созданы и добавлены в FAISS индекс")

        # Сохраняем метаданные и маппинги (остальной код без изменений)
        added_ids = []
        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            added_ids.append(faiss_id)

            # Сохраняем метаданные
            self.metadata[chunk.chunk_id] = {
                'text': chunk.text,  # ❗ Сохраняем ИСХОДНЫЙ текст, не расширенный
                'source_file': chunk.source_file,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata,
                'added_date': datetime.now().isoformat(),
                'faiss_id': faiss_id
            }

            # Обновляем маппинги
            self.id_to_chunk_id[faiss_id] = chunk.chunk_id
            self.chunk_id_to_id[chunk.chunk_id] = faiss_id

        final_message = f"✅ Добавлено {len(chunks)} чанков в индекс с векторизацией метаданных. Всего в индексе: {self.index.ntotal}"
        # print(final_message)
        logger.info(final_message)

        return added_ids

    def _add_chunks_legacy(self, chunks: List[TextChunk]) -> List[int]:
        """Старая логика добавления чанков (для совместимости)"""
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
                'faiss_id': faiss_id,
                'has_visual_vector': False  # ✅ Добавлено для совместимости
            }

            # Обновляем маппинги
            self.id_to_chunk_id[faiss_id] = chunk.chunk_id
            self.chunk_id_to_id[chunk.chunk_id] = faiss_id

        logger.info(f"Добавлено {len(chunks)} чанков в индекс. Всего в индексе: {self.index.ntotal}")
        return added_ids

    # ✅ НОВЫЕ методы для мультимодального режима

    def add_text_chunk(self, chunk: TextChunk) -> int:
        """Добавляет текстовый чанк в мультимодальный индекс"""
        if not self.enable_visual_search:
            # Fallback на старую логику
            return self._add_chunks_legacy([chunk])[0]

        if self.text_index is None:
            self.create_index()

        # 🔧 ИСПРАВЛЯЕМ: Добавляем метаданные к тексту для векторизации
        metadata = chunk.metadata
        full_text_parts = [chunk.text]

        # Добавляем важные метаданные к тексту для векторизации
        if metadata.get('title'):
            full_text_parts.append(f"Заголовок: {metadata['title']}")
        if metadata.get('description'):
            full_text_parts.append(f"Описание: {metadata['description']}")
        if metadata.get('category'):
            full_text_parts.append(f"Категория: {metadata['category']}")
        if metadata.get('parent'):
            full_text_parts.append(f"Раздел: {metadata['parent']}")

        combined_text = " ".join(full_text_parts)

        # 🔧 DEBUG: Показываем что векторизуем (принудительный вывод)
        debug_msg = f"🔧 ТЕКСТ-ТОЛЬКО {chunk.source_file} (чанк {chunk.chunk_index}):"
        # print(debug_msg)
        logger.info(debug_msg)

        size_msg = f"   📝 Исходный: {len(chunk.text)} символов → Расширенный: {len(combined_text)} символов"
        # print(size_msg)
        logger.info(size_msg)

        meta_msg = f"   🏷️ title='{metadata.get('title', 'НЕТ')}' | description='{metadata.get('description', 'НЕТ')}' | category='{metadata.get('category', 'НЕТ')}'"
        # print(meta_msg)
        logger.info(meta_msg)

        # Создаем текстовый embedding из РАСШИРЕННОГО текста
        text_embedding = self.create_embeddings([combined_text])

        # Добавляем в текстовый индекс
        text_faiss_id = self.text_index.ntotal
        self.text_index.add(text_embedding)

        # Сохраняем метаданные
        self.metadata[chunk.chunk_id] = {
            'text': chunk.text,  # ❗ Сохраняем ИСХОДНЫЙ текст, не расширенный
            'source_file': chunk.source_file,
            'chunk_index': chunk.chunk_index,
            'metadata': chunk.metadata,
            'added_date': datetime.now().isoformat(),
            'text_faiss_id': text_faiss_id,
            'visual_faiss_id': None,
            'has_visual_vector': False
        }

        # Обновляем маппинги
        self.text_id_to_chunk_id[text_faiss_id] = chunk.chunk_id
        self.chunk_id_to_ids[chunk.chunk_id] = {'text_id': text_faiss_id, 'visual_id': None}

        success_msg = f"✅ Текстовый чанк добавлен: текст_id={text_faiss_id}"
        # print(success_msg)
        logger.info(success_msg)

        return text_faiss_id
    def add_multimodal_chunk(self, chunk: TextChunk, visual_vector: np.ndarray) -> Tuple[int, int]:
        """
        Добавляет мультимодальный чанк (текст + изображение)

        Args:
            chunk: Текстовый чанк с описанием изображения
            visual_vector: Визуальный вектор изображения

        Returns:
            Tuple[int, int]: (text_faiss_id, visual_faiss_id)
        """
        if not self.enable_visual_search:
            raise ValueError("Мультимодальные чанки требуют enable_visual_search=True")

        if self.text_index is None or self.visual_index is None:
            self.create_index()

        # 1. 🔧 ИСПРАВЛЕННАЯ текстовая часть с метаданными
        metadata = chunk.metadata
        full_text_parts = [chunk.text]

        # Добавляем важные метаданные к тексту для векторизации
        if metadata.get('title'):
            full_text_parts.append(f"Заголовок: {metadata['title']}")
        if metadata.get('description'):
            full_text_parts.append(f"Описание: {metadata['description']}")
        if metadata.get('category'):
            full_text_parts.append(f"Категория: {metadata['category']}")
        if metadata.get('parent'):
            full_text_parts.append(f"Раздел: {metadata['parent']}")

        combined_text = " ".join(full_text_parts)

        # 🔧 DEBUG: Показываем что векторизуем (принудительный вывод)
        debug_msg = f"🔧 МУЛЬТИМОДАЛ {chunk.source_file} (чанк {chunk.chunk_index}):"
        # print(debug_msg)
        logger.info(debug_msg)

        size_msg = f"   📝 Исходный: {len(chunk.text)} символов → Расширенный: {len(combined_text)} символов"
        # print(size_msg)
        logger.info(size_msg)

        meta_msg = f"   🏷️ title='{metadata.get('title', 'НЕТ')}' | description='{metadata.get('description', 'НЕТ')}' | category='{metadata.get('category', 'НЕТ')}'"
        # print(meta_msg)
        logger.info(meta_msg)

        # Векторизуем РАСШИРЕННЫЙ текст
        text_embedding = self.create_embeddings([combined_text])
        text_faiss_id = self.text_index.ntotal
        self.text_index.add(text_embedding)

        # 2. Добавляем визуальную часть (без изменений)
        # Нормализуем визуальный вектор
        if self.index_type == "FlatIP":
            normalized_visual = visual_vector / np.linalg.norm(visual_vector)
        else:
            normalized_visual = visual_vector

        visual_faiss_id = self.visual_index.ntotal
        self.visual_index.add(normalized_visual.reshape(1, -1))

        # 3. Сохраняем метаданные
        self.metadata[chunk.chunk_id] = {
            'text': chunk.text,  # ❗ Сохраняем ИСХОДНЫЙ текст, не расширенный
            'source_file': chunk.source_file,
            'chunk_index': chunk.chunk_index,
            'metadata': chunk.metadata,
            'added_date': datetime.now().isoformat(),
            'text_faiss_id': text_faiss_id,
            'visual_faiss_id': visual_faiss_id,
            'has_visual_vector': True
        }

        # 4. Обновляем маппинги
        self.text_id_to_chunk_id[text_faiss_id] = chunk.chunk_id
        self.visual_id_to_chunk_id[visual_faiss_id] = chunk.chunk_id
        self.chunk_id_to_ids[chunk.chunk_id] = {
            'text_id': text_faiss_id,
            'visual_id': visual_faiss_id
        }

        success_msg = f"✅ Мультимодальный чанк добавлен: текст_id={text_faiss_id}, визуал_id={visual_faiss_id}"
        # print(success_msg)
        logger.info(success_msg)

        return text_faiss_id, visual_faiss_id
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Поиск похожих документов (совместимость со старым API)"""
        if not self.enable_visual_search:
            # Старая логика
            return self._search_legacy(query, k, score_threshold)
        else:
            # В мультимодальном режиме ищем по текстовому индексу
            return self.search_text(query, k, score_threshold)

    def _search_legacy(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Старая логика поиска (для совместимости)"""
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

    # ✅ НОВЫЕ методы поиска для мультимодального режима

    def search_text(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Текстовый поиск в мультимодальном режиме"""
        if not self.enable_visual_search or self.text_index is None or self.text_index.ntotal == 0:
            return []

        query_embedding = self.create_embeddings([query])
        scores, indices = self.text_index.search(query_embedding, k)

        return self._format_search_results(scores[0], indices[0], "text", score_threshold)

    def search_visual(self, visual_query: np.ndarray, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Визуальный поиск"""
        if not self.enable_visual_search or self.visual_index is None or self.visual_index.ntotal == 0:
            return []

        # Нормализуем запрос
        if self.index_type == "FlatIP":
            normalized_query = visual_query / np.linalg.norm(visual_query)
        else:
            normalized_query = visual_query

        scores, indices = self.visual_index.search(normalized_query.reshape(1, -1), k)

        return self._format_search_results(scores[0], indices[0], "visual", score_threshold)

    def search_multimodal(self, text_query: str = None, visual_query: np.ndarray = None,
                          k: int = 5, text_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Комбинированный мультимодальный поиск

        Args:
            text_query: Текстовый запрос
            visual_query: Визуальный вектор запроса
            k: Количество результатов
            text_weight: Вес текстового поиска (0.0-1.0)

        Returns:
            List[Dict]: Объединенные результаты
        """
        if not self.enable_visual_search:
            # Fallback на обычный текстовый поиск
            return self.search_text(text_query, k) if text_query else []

        all_results = {}  # {chunk_id: {'text_score': X, 'visual_score': Y}}

        # Текстовый поиск
        if text_query:
            text_results = self.search_text(text_query, k=k * 2)  # Берем больше для объединения
            for result in text_results:
                chunk_id = result['chunk_id']
                all_results[chunk_id] = all_results.get(chunk_id, {})
                all_results[chunk_id]['text_score'] = result['score']
                all_results[chunk_id]['result_data'] = result

        # Визуальный поиск
        if visual_query is not None:
            visual_results = self.search_visual(visual_query, k=k * 2)
            for result in visual_results:
                chunk_id = result['chunk_id']
                all_results[chunk_id] = all_results.get(chunk_id, {})
                all_results[chunk_id]['visual_score'] = result['score']
                if 'result_data' not in all_results[chunk_id]:
                    all_results[chunk_id]['result_data'] = result

        # Вычисляем комбинированный score
        combined_results = []
        for chunk_id, scores in all_results.items():
            text_score = scores.get('text_score', 0.0)
            visual_score = scores.get('visual_score', 0.0)

            # Комбинированный score
            if 'text_score' in scores and 'visual_score' in scores:
                # Оба типа поиска нашли этот чанк
                combined_score = text_weight * text_score + (1 - text_weight) * visual_score
                search_type = "multimodal"
            elif 'text_score' in scores:
                combined_score = text_score * text_weight
                search_type = "text_only"
            else:
                combined_score = visual_score * (1 - text_weight)
                search_type = "visual_only"

            result = scores['result_data'].copy()
            result['combined_score'] = combined_score
            result['search_type'] = search_type
            result['text_score'] = text_score
            result['visual_score'] = visual_score

            combined_results.append(result)

        # Сортируем по комбинированному score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

        return combined_results[:k]

    def _format_search_results(self, scores: np.ndarray, indices: np.ndarray,
                               search_type: str, score_threshold: float) -> List[Dict[str, Any]]:
        """Форматирует результаты поиска"""
        results = []

        if search_type == "text":
            id_mapping = self.text_id_to_chunk_id
        elif search_type == "visual":
            id_mapping = self.visual_id_to_chunk_id
        else:
            id_mapping = self.id_to_chunk_id  # Fallback

        for score, idx in zip(scores, indices):
            if idx == -1 or score < score_threshold:
                continue

            chunk_id = id_mapping.get(idx)
            if chunk_id and chunk_id in self.metadata:
                result = {
                    'chunk_id': chunk_id,
                    'score': float(score),
                    'search_type': search_type,
                    'text': self.metadata[chunk_id]['text'],
                    'source_file': self.metadata[chunk_id]['source_file'],
                    'metadata': self.metadata[chunk_id]['metadata']
                }
                results.append(result)

        return results

    # ✅ ОБНОВЛЕННЫЕ методы сохранения/загрузки

    def save_index(self):
        """Сохраняет индекс и метаданные на диск"""
        logger.info("Сохраняем FAISS индекс(ы) и метаданные")

        if self.enable_visual_search:
            # Сохраняем мультимодальные индексы
            if self.text_index is not None:
                faiss.write_index(self.text_index, str(self.text_index_path))
                logger.info(f"✅ Текстовый индекс сохранен: {self.text_index.ntotal} векторов")

            if self.visual_index is not None:
                faiss.write_index(self.visual_index, str(self.visual_index_path))
                logger.info(f"✅ Визуальный индекс сохранен: {self.visual_index.ntotal} векторов")
        else:
            # Сохраняем единый индекс
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
                logger.info(f"✅ Индекс сохранен: {self.index.ntotal} векторов")

        # Сохраняем метаданные
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        # Сохраняем маппинги
        if self.enable_visual_search:
            mappings_data = {
                'text_id_to_chunk_id': {str(k): v for k, v in self.text_id_to_chunk_id.items()},
                'visual_id_to_chunk_id': {str(k): v for k, v in self.visual_id_to_chunk_id.items()},
                'chunk_id_to_ids': self.chunk_id_to_ids
            }
        else:
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
            'enable_visual_search': self.enable_visual_search,  # ✅ НОВОЕ
            'text_dimension': self.text_dimension,  # ✅ НОВОЕ
            'visual_dimension': self.visual_dimension,  # ✅ НОВОЕ
            'total_vectors': self._get_total_vectors(),
            'text_vectors': self.text_index.ntotal if self.text_index else 0,  # ✅ НОВОЕ
            'visual_vectors': self.visual_index.ntotal if self.visual_index else 0,  # ✅ НОВОЕ
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    def _get_total_vectors(self) -> int:
        """Возвращает общее количество векторов"""
        if self.enable_visual_search:
            text_count = self.text_index.ntotal if self.text_index else 0
            visual_count = self.visual_index.ntotal if self.visual_index else 0
            return text_count + visual_count
        else:
            return self.index.ntotal if self.index else 0

    def load_index(self) -> bool:


        """Загружает индекс и метаданные с диска"""
        required_paths = [self.metadata_path, self.mappings_path, self.config_path]

        missing_files = [str(p) for p in required_paths if not p.exists()]
        if missing_files:
            logger.warning(f"Отсутствуют файлы индекса: {', '.join(missing_files)}")
            return False

        try:
            logger.info("Загружаем FAISS индекс(ы) и метаданные")

            # Загружаем конфигурацию
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.model_name = config['model_name']
            self.index_type = config['index_type']
            self.dimension = config['dimension']
            # ✅ НОВЫЕ параметры конфигурации
            self.enable_visual_search = config.get('enable_visual_search', False)
            self.text_dimension = config.get('text_dimension', self.dimension)
            self.visual_dimension = config.get('visual_dimension', settings.VISUAL_EMBEDDING_DIMENSION)

            if self.enable_visual_search:
                # Загружаем мультимодальные индексы
                if self.text_index_path.exists():
                    self.text_index = faiss.read_index(str(self.text_index_path))
                    logger.info(f"✅ Текстовый индекс загружен: {self.text_index.ntotal} векторов")

                if self.visual_index_path.exists():
                    self.visual_index = faiss.read_index(str(self.visual_index_path))
                    logger.info(f"✅ Визуальный индекс загружен: {self.visual_index.ntotal} векторов")
            else:
                # Загружаем единый индекс
                if self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(f"✅ Индекс загружен: {self.index.ntotal} векторов")

            # Загружаем метаданные
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            # Загружаем маппинги
            with open(self.mappings_path, 'r', encoding='utf-8') as f:
                mappings_data = json.load(f)

            if self.enable_visual_search:
                # Мультимодальные маппинги
                self.text_id_to_chunk_id = {int(k): v for k, v in mappings_data.get('text_id_to_chunk_id', {}).items()}
                self.visual_id_to_chunk_id = {int(k): v for k, v in
                                              mappings_data.get('visual_id_to_chunk_id', {}).items()}
                self.chunk_id_to_ids = mappings_data.get('chunk_id_to_ids', {})
            else:
                # Старые маппинги
                self.id_to_chunk_id = {int(k): v for k, v in mappings_data.get('id_to_chunk_id', {}).items()}
                self.chunk_id_to_id = mappings_data.get('chunk_id_to_id', {})

            logger.info(f"✅ Метаданные загружены: {len(self.metadata)} чанков")
            return True

        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {e}")
            return False

    def get_index_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику индекса"""
        if self.enable_visual_search:
            return self._get_multimodal_statistics()
        else:
            return self._get_legacy_statistics()

    def _get_legacy_statistics(self) -> Dict[str, Any]:
        """Старая статистика (для совместимости)"""
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
            'is_trained': getattr(self.index, 'is_trained', True),
            'enable_visual_search': False
        }

    def _get_multimodal_statistics(self) -> Dict[str, Any]:
        """Мультимодальная статистика"""
        stats = {
            'status': 'ready' if (self.text_index is not None or self.visual_index is not None) else 'not_initialized',
            'client_id': self.client_id,
            'enable_visual_search': True,
            'text_dimension': self.text_dimension,
            'visual_dimension': self.visual_dimension,
            'text_vectors_count': self.text_index.ntotal if self.text_index else 0,
            'visual_vectors_count': self.visual_index.ntotal if self.visual_index else 0,
            'total_chunks': len(self.metadata),
            'multimodal_chunks': sum(1 for chunk in self.metadata.values()
                                     if chunk.get('has_visual_vector', False))
        }

        if stats['status'] == 'ready':
            # Дополнительная статистика
            sources_stats = {}
            categories = {}
            file_types = {}
            visual_content_count = 0

            for chunk_data in self.metadata.values():
                metadata = chunk_data.get('metadata', {})

                # Источники
                source = chunk_data.get('source_file', 'unknown')
                sources_stats[source] = sources_stats.get(source, 0) + 1

                # Типы файлов
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1

                # Категории
                category = metadata.get('category', 'uncategorized')
                categories[category] = categories.get(category, 0) + 1

                # Визуальный контент
                if chunk_data.get('has_visual_vector', False):
                    visual_content_count += 1

            stats.update({
                'sources_count': len(sources_stats),
                'sources_distribution': sources_stats,
                'file_types_distribution': file_types,
                'categories_distribution': categories,
                'visual_content_ratio': visual_content_count / len(self.metadata) if self.metadata else 0
            })

        return stats

    # Методы для обратной совместимости
    def get_index_stats(self) -> Dict[str, Any]:
        """Алиас для get_index_statistics (совместимость)"""
        return self.get_index_statistics()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Возвращает все чанки"""
        return list(self.metadata.values())

    def get_chunks_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Возвращает все чанки из определенного источника"""
        return [metadata for metadata in self.metadata.values()
                if metadata['source_file'] == source_file]

    def remove_chunks(self, chunk_ids: List[str]) -> bool:
        """Удаляет чанки из индекса (упрощенная версия)"""
        removed_count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.metadata:
                # Удаляем из метаданных
                del self.metadata[chunk_id]
                removed_count += 1

                # Удаляем из маппингов
                if self.enable_visual_search:
                    # Мультимодальные маппинги
                    if chunk_id in self.chunk_id_to_ids:
                        ids_info = self.chunk_id_to_ids[chunk_id]

                        if ids_info.get('text_id') is not None:
                            self.text_id_to_chunk_id.pop(ids_info['text_id'], None)

                        if ids_info.get('visual_id') is not None:
                            self.visual_id_to_chunk_id.pop(ids_info['visual_id'], None)

                        del self.chunk_id_to_ids[chunk_id]
                else:
                    # Старые маппинги
                    if chunk_id in self.chunk_id_to_id:
                        faiss_id = self.chunk_id_to_id[chunk_id]
                        self.id_to_chunk_id.pop(faiss_id, None)
                        del self.chunk_id_to_id[chunk_id]

        if removed_count > 0:
            logger.warning(
                f"Удалены метаданные для {removed_count} чанков. Для полной очистки требуется пересоздание индекса")

        return removed_count > 0

    def clear_index(self):
        """Очищает индекс и все метаданные"""
        logger.warning(f"Очищаем все данные из индекса клиента {self.client_id}")

        # Очищаем индексы
        self.index = None
        self.text_index = None
        self.visual_index = None

        # Очищаем метаданные
        self.metadata = {}
        self.id_to_chunk_id = {}
        self.chunk_id_to_id = {}
        self.text_id_to_chunk_id = {}
        self.visual_id_to_chunk_id = {}
        self.chunk_id_to_ids = {}

        # Удаляем файлы с диска
        for path in [self.index_path, self.text_index_path, self.visual_index_path,
                     self.metadata_path, self.mappings_path, self.config_path]:
            if path.exists():
                path.unlink()

    # ✅ НОВЫЕ методы для удобства работы с визуальными векторами

    def get_visual_vector(self, chunk_id: str) -> Optional[np.ndarray]:
        """Получает визуальный вектор чанка"""
        if not self.enable_visual_search or chunk_id not in self.metadata:
            return None

        chunk_data = self.metadata[chunk_id]
        visual_faiss_id = chunk_data.get('visual_faiss_id')

        if visual_faiss_id is not None and self.visual_index is not None:
            try:
                return self.visual_index.reconstruct(visual_faiss_id)
            except Exception as e:
                logger.error(f"Ошибка получения визуального вектора для {chunk_id}: {e}")

        return None

    def get_similar_visual_chunks(self, chunk_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Находит визуально похожие чанки на существующий"""
        visual_vector = self.get_visual_vector(chunk_id)
        if visual_vector is None:
            return []

        # Ищем похожие (исключаем сам чанк)
        results = self.search_visual(visual_vector, k=k + 1)

        # Убираем исходный чанк из результатов
        filtered_results = []
        for result in results:
            if result.get('chunk_id') != chunk_id:
                filtered_results.append(result)

        return filtered_results[:k]

    def export_visual_vectors(self) -> Dict[str, Any]:
        """Экспортирует все визуальные векторы"""
        if not self.enable_visual_search or self.visual_index is None:
            return {'error': 'Визуальные векторы недоступны'}

        export_data = {
            'client_id': self.client_id,
            'visual_vectors_count': self.visual_index.ntotal,
            'visual_dimension': self.visual_dimension,
            'vectors': [],
            'metadata': []
        }

        for chunk_data in self.metadata.values():
            if chunk_data.get('has_visual_vector', False):
                visual_id = chunk_data.get('visual_faiss_id')
                if visual_id is not None:
                    try:
                        vector = self.visual_index.reconstruct(visual_id)
                        export_data['vectors'].append(vector.tolist())
                        export_data['metadata'].append({
                            'chunk_id': chunk_data.get('chunk_id'),
                            'source_file': chunk_data.get('source_file'),
                            'visual_faiss_id': visual_id
                        })
                    except Exception as e:
                        logger.error(f"Ошибка экспорта вектора {visual_id}: {e}")

        return export_data


# Функция для проверки совместимости и миграции
def check_index_compatibility(client_id: str) -> Dict[str, Any]:
    """Проверяет совместимость существующего индекса"""
    client_dir = settings.CLIENTS_DIR / client_id

    compatibility = {
        'client_id': client_id,
        'has_legacy_index': (client_dir / "index.faiss").exists(),
        'has_multimodal_index': (client_dir / "text_index.faiss").exists() or (
                    client_dir / "visual_index.faiss").exists(),
        'config_exists': (client_dir / "config.json").exists(),
        'migration_needed': False,
        'recommendations': []
    }

    if compatibility['has_legacy_index'] and not compatibility['has_multimodal_index']:
        compatibility['migration_needed'] = True
        compatibility['recommendations'].append("Можно мигрировать на мультимодальную архитектуру")

    if compatibility['has_multimodal_index']:
        compatibility['recommendations'].append("Мультимодальная архитектура уже настроена")

    if not compatibility['config_exists']:
        compatibility['recommendations'].append("Отсутствует файл конфигурации")

    return compatibility


# Функция для создания менеджера с автоопределением режима
def create_faiss_manager(client_id: str, enable_visual_search: bool = None) -> FAISSManager:
    """
    Создает FAISS менеджер с автоопределением режима

    Args:
        client_id: ID клиента
        enable_visual_search: Принудительно включить/выключить визуальный поиск.
                             None = автоопределение по существующей конфигурации

    Returns:
        FAISSManager: Настроенный менеджер
    """
    # Автоопределение режима
    if enable_visual_search is None:
        client_dir = settings.CLIENTS_DIR / client_id
        config_path = client_dir / "config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                enable_visual_search = config.get('enable_visual_search', False)
                logger.info(f"Автоопределение режима для {client_id}: визуальный_поиск={enable_visual_search}")
            except Exception:
                enable_visual_search = settings.ENABLE_VISUAL_SEARCH_BY_DEFAULT
        else:
            enable_visual_search = settings.ENABLE_VISUAL_SEARCH_BY_DEFAULT

    return FAISSManager(client_id=client_id, enable_visual_search=enable_visual_search)