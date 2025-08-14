import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import numpy as np

from .data.loaders import DocumentLoader, DocumentParser
from .data.chunkers import DocumentChunker, TextChunk
from .data.image_processor import ImageProcessor, MultiModalProcessor
from .vectorstore.faiss_manager import FAISSManager, create_faiss_manager
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Главный класс для обработки документов и управления RAG пайплайном с поддержкой мультимодальности"""

    def __init__(self, client_id: str, enable_visual_search: bool = None):  # ✅ ДОБАВЛЕН параметр
        """
        Инициализация процессора документов

        Args:
            client_id: ID клиента
            enable_visual_search: Включить визуальный поиск. None = автоопределение
        """
        self.client_id = client_id

        # ✅ НОВАЯ логика определения режима
        if enable_visual_search is None:
            self.enable_visual_search = settings.ENABLE_VISUAL_SEARCH_BY_DEFAULT
        else:
            self.enable_visual_search = enable_visual_search

        logger.info(
            f"Инициализация DocumentProcessor для клиента {client_id}, визуальный_поиск={self.enable_visual_search}")

        # ✅ ОБНОВЛЕННАЯ инициализация компонентов
        self.document_loader = DocumentLoader(client_id=client_id)  # Передаем client_id
        self.document_parser = DocumentParser()
        self.chunker = DocumentChunker()

        # ✅ НОВЫЕ процессоры изображений
        if self.enable_visual_search:
            try:
                self.multimodal_processor = MultiModalProcessor()
                logger.info("🖼️ Мультимодальный режим включен - поддержка визуального поиска")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось инициализировать мультимодальный процессор: {e}")
                logger.info("📝 Переключение на текстовый режим")
                self.enable_visual_search = False
                self.image_processor = ImageProcessor()
        else:
            self.image_processor = ImageProcessor()
            logger.info("📝 Текстовый режим - только текстовое описание изображений")

        # ✅ ОБНОВЛЕННЫЙ FAISS менеджер
        self.faiss_manager = create_faiss_manager(client_id, self.enable_visual_search)

        # Загружаем существующий индекс при инициализации
        self.faiss_manager.load_index()

    def process_documents_from_json(self, json_file_path: str,
                                    update_existing: bool = False) -> Dict[str, Any]:
        """
        Обрабатывает документы из JSON файла с поддержкой мультимодальности

        Args:
            json_file_path: Путь к JSON файлу со списком документов
            update_existing: Обновлять ли существующие документы

        Returns:
            Статистика обработки
        """
        logger.info(
            f"Начинаем {'мультимодальную' if self.enable_visual_search else 'текстовую'} обработку документов из {json_file_path}")

        # ✅ РАСШИРЕННАЯ статистика
        stats = {
            'total_documents': 0,
            'downloaded': 0,
            'parsed': 0,
            'chunked': 0,
            'indexed': 0,
            'images_processed': 0,  # ✅ НОВОЕ
            'visual_vectors_created': 0,  # ✅ НОВОЕ
            'text_vectors_created': 0,  # ✅ НОВОЕ
            'errors': [],
            'processed_files': [],
            'start_time': datetime.now().isoformat(),
            'multimodal_mode': self.enable_visual_search  # ✅ НОВОЕ
        }

        try:
            # Загружаем и скачиваем документы
            documents_info = self.document_loader.process_json_documents(json_file_path)
            stats['total_documents'] = len(documents_info)

            all_chunks = []

            for doc_info in documents_info:
                try:
                    file_path = doc_info['file_path']
                    url = doc_info['url']
                    original_metadata = doc_info['metadata']

                    logger.info(f"Обрабатываем: {file_path.name}")

                    # Проверяем, нужно ли обновлять существующие чанки
                    if not update_existing:
                        existing_chunks = self.faiss_manager.get_chunks_by_source(file_path.name)
                        if existing_chunks:
                            logger.info(f"Документ {file_path.name} уже существует, пропускаем")
                            continue

                    stats['downloaded'] += 1

                    # ✅ НОВАЯ логика обработки в зависимости от типа файла
                    is_image = self._is_image_file(file_path)

                    if is_image:
                        # Обработка изображения
                        text, visual_vector, enhanced_metadata = self._process_image(
                            file_path, url, original_metadata
                        )
                        stats['images_processed'] += 1

                        if self.enable_visual_search and visual_vector is not None:
                            stats['visual_vectors_created'] += 1

                    else:
                        # Обработка документа (как раньше)
                        text, doc_metadata = self.document_parser.parse_document(file_path)
                        visual_vector = None

                        enhanced_metadata = self._create_enhanced_metadata(
                            original_metadata, url, file_path, doc_metadata
                        )

                    if not text.strip():
                        stats['errors'].append(f"Пустой текст в {file_path.name}")
                        continue

                    stats['parsed'] += 1

                    # Создаем чанки
                    chunks = self.chunker.create_chunks(text, file_path.name, enhanced_metadata)

                    if chunks:
                        # ✅ НОВАЯ логика добавления чанков
                        for chunk in chunks:
                            if visual_vector is not None and self.enable_visual_search:
                                # Мультимодальный чанк
                                self.faiss_manager.add_multimodal_chunk(chunk, visual_vector)
                            else:
                                # Обычный текстовый чанк (старая логика)
                                if self.enable_visual_search:
                                    self.faiss_manager.add_text_chunk(chunk)
                                else:
                                    # Полная совместимость со старым API
                                    pass  # Будет добавлен через add_chunks ниже

                        if not self.enable_visual_search:
                            # Старая логика для полной совместимости
                            self.faiss_manager.add_chunks(chunks)

                        all_chunks.extend(chunks)
                        stats['chunked'] += len(chunks)
                        stats['text_vectors_created'] += len(chunks)

                        stats['processed_files'].append({
                            'filename': file_path.name,
                            'file_type': 'image' if is_image else 'document',
                            'chunks_count': len(chunks),
                            'characters': len(text),
                            'has_visual_vector': visual_vector is not None,
                            'url': url,
                            'metadata_keys': list(enhanced_metadata.keys())
                        })

                        logger.info(
                            f"✅ {file_path.name}: {len(chunks)} чанков, визуальный={'да' if visual_vector is not None else 'нет'}")

                    # ✅ НОВОЕ: Удаляем временный файл после обработки
                    if not settings.KEEP_DOWNLOADED_FILES:
                        try:
                            file_path.unlink()
                            logger.debug(f"🗑️ Файл {file_path.name} удален после обработки")
                        except Exception as e:
                            logger.warning(f"⚠️ Не удалось удалить файл {file_path.name}: {e}")

                except Exception as e:
                    error_msg = f"Ошибка при обработке {doc_info.get('file_path', 'unknown')}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
                    continue

            # Сохраняем индекс
            if all_chunks:
                self.faiss_manager.save_index()
                stats['indexed'] = len(all_chunks)
                logger.info(
                    f"✅ Индексы сохранены: {stats['text_vectors_created']} текстовых, {stats['visual_vectors_created']} визуальных")

            stats['end_time'] = datetime.now().isoformat()
            stats['success'] = True

            # ✅ НОВОЕ: Очищаем GPU память если использовали CLIP
            if self.enable_visual_search and hasattr(self, 'multimodal_processor'):
                self.multimodal_processor.cleanup_gpu_memory()

        except Exception as e:
            stats['end_time'] = datetime.now().isoformat()
            stats['success'] = False
            stats['errors'].append(f"Критическая ошибка: {e}")
            logger.error(f"Критическая ошибка при обработке: {e}")

        return stats

    # ✅ НОВЫЕ вспомогательные методы

    def _is_image_file(self, file_path: Path) -> bool:
        """Проверяет, является ли файл изображением"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return file_path.suffix.lower() in image_extensions

    def _process_image(self, file_path: Path, url: str, original_metadata: Dict) -> tuple[
        str, Optional[np.ndarray], Dict]:
        """Обрабатывает изображение в зависимости от режима"""

        if self.enable_visual_search and hasattr(self, 'multimodal_processor'):
            # Мультимодальная обработка
            text, visual_vector, updated_metadata = self.multimodal_processor.process_image_multimodal(
                file_path, original_metadata
            )

            # Добавляем базовые метаданные
            enhanced_metadata = self._create_enhanced_metadata(
                updated_metadata, url, file_path, None
            )

            return text, visual_vector, enhanced_metadata

        else:
            # Только текстовая обработка (как раньше)
            text = self.image_processor.create_image_embedding_text(file_path, original_metadata)

            enhanced_metadata = self._create_enhanced_metadata(
                original_metadata, url, file_path, None
            )

            return text, None, enhanced_metadata

    # В document_processor.py в функции _create_enhanced_metadata ЗАМЕНИТЕ:

    def _create_enhanced_metadata(self, original_metadata: Dict, url: str,
                                  file_path: Path, doc_metadata: Any) -> Dict[str, Any]:
        """Создает расширенные метаданные"""

        enhanced_metadata = {}

        # Сначала копируем ВСЁ из original_metadata
        enhanced_metadata.update(original_metadata)

        # Потом дополняем техническими полями
        enhanced_metadata.update({
            'file_type': file_path.suffix.lower(),
            'filename': file_path.name,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'processing_date': datetime.now().isoformat(),
            'client_id': self.client_id,
        })

        # Дополняем из doc_metadata, но только если в enhanced_metadata нет значения или оно пустое
        if doc_metadata:
            for key, value in doc_metadata.__dict__.items():
                if not enhanced_metadata.get(key) and value not in (None, ''):
                    enhanced_metadata[key] = value

        # Гарантируем, что важные поля всегда есть
        if not enhanced_metadata.get('source_url'):
            enhanced_metadata['source_url'] = url

        if not enhanced_metadata.get('category'):
            enhanced_metadata['category'] = enhanced_metadata.get('parent', 'uncategorized')

        if not enhanced_metadata.get('title'):
            enhanced_metadata['title'] = enhanced_metadata.get('description', file_path.stem)

        # Отладка
        logger.info(f"🔧 DEBUG enhanced_metadata результат для {file_path.name}:")
        logger.info(f"   enhanced_metadata['source_url']: '{enhanced_metadata.get('source_url')}'")
        logger.info(f"   enhanced_metadata['description']: '{enhanced_metadata.get('description')}'")
        logger.info(f"   enhanced_metadata['title']: '{enhanced_metadata.get('title')}'")
        logger.info(f"   enhanced_metadata keys count: {len(enhanced_metadata)}")

        return enhanced_metadata

    def search_documents(self, query: str = None, k: int = 5,
                         min_score: float = 0.0,
                         filters: Optional[Dict[str, Any]] = None,
                         search_mode: str = "auto") -> List[Dict[str, Any]]:
        """
        Универсальный поиск документов с поддержкой мультимодальности

        Args:
            query: Текстовый запрос
            k: Количество результатов
            min_score: Минимальный score для результатов
            filters: Фильтры по метаданным (например, {'category': 'tech'})
            search_mode: Режим поиска ("auto", "text", "visual_description")

        Returns:
            Список найденных документов с релевантностью
        """
        if not query:
            return []

        # Определяем режим поиска
        if search_mode == "auto":
            if self.enable_visual_search:
                # В мультимодальном режиме по умолчанию ищем по тексту
                search_mode = "text"
            else:
                search_mode = "text"

        # Выполняем поиск
        if search_mode == "text":
            results = self.faiss_manager.search(query, k=k * 2, score_threshold=min_score)
        elif search_mode == "visual_description" and self.enable_visual_search:
            # Поиск изображений по текстовому описанию через CLIP
            results = self.search_by_text_description(query, k=k * 2, search_images_only=False)
        else:
            # Fallback на обычный текстовый поиск
            results = self.faiss_manager.search(query, k=k * 2, score_threshold=min_score)

        # Применяем фильтры
        if filters:
            results = self._apply_filters(results, filters)

        return results[:k]

    def search_similar_images(self, query_image_path: Union[str, Path], k: int = 5,
                              min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Поиск визуально похожих изображений

        Args:
            query_image_path: Путь к изображению-запросу
            k: Количество результатов
            min_score: Минимальный score сходства

        Returns:
            List[Dict]: Список похожих изображений с scores
        """
        if not self.enable_visual_search:
            logger.warning("Визуальный поиск отключен. Включите enable_visual_search=True")
            return []

        if not hasattr(self, 'multimodal_processor'):
            logger.error("Мультимодальный процессор недоступен")
            return []

        try:
            # Создаем визуальный запрос
            query_path = Path(query_image_path)
            visual_query = self.multimodal_processor.create_visual_embedding(query_path)

            # Ищем похожие
            results = self.faiss_manager.search_visual(visual_query, k=k, score_threshold=min_score)

            logger.info(f"🔍 Визуальный поиск по {query_path.name}: найдено {len(results)} результатов")
            return results

        except Exception as e:
            logger.error(f"Ошибка визуального поиска: {e}")
            return []

    def search_by_text_description(self, text_description: str, k: int = 5,
                                   search_images_only: bool = True) -> List[Dict[str, Any]]:
        """
        Поиск изображений по текстовому описанию через CLIP

        Args:
            text_description: Описание искомого изображения ("фасад здания")
            k: Количество результатов
            search_images_only: Искать только среди изображений

        Returns:
            List[Dict]: Результаты поиска
        """
        if not self.enable_visual_search:
            # Fallback на обычный текстовый поиск
            return self.search_documents(text_description, k=k)

        if not hasattr(self, 'multimodal_processor'):
            logger.error("Мультимодальный процессор недоступен")
            return self.search_documents(text_description, k=k)

        try:
            # Создаем визуальный запрос из текста
            visual_query = self.multimodal_processor.search_by_text_description(text_description)

            # Ищем в визуальном пространстве
            results = self.faiss_manager.search_visual(visual_query, k=k * 2)

            # Фильтруем только изображения
            if search_images_only:
                image_results = []
                for result in results:
                    metadata = result.get('metadata', {})
                    if metadata.get('is_image', False):
                        image_results.append(result)
                results = image_results[:k]

            logger.info(f"🎯 Поиск изображений по описанию '{text_description}': найдено {len(results)}")
            return results

        except Exception as e:
            logger.error(f"Ошибка поиска по описанию: {e}")
            return []

    def search_multimodal(self, text_query: str = None, image_query_path: Union[str, Path] = None,
                          k: int = 5, text_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        Комбинированный мультимодальный поиск

        Args:
            text_query: Текстовый запрос
            image_query_path: Путь к изображению-запросу
            k: Количество результатов
            text_weight: Вес текстового поиска (0.0-1.0)

        Returns:
            List[Dict]: Объединенные результаты поиска
        """
        if not self.enable_visual_search:
            logger.warning("Мультимодальный поиск недоступен без visual_search")
            return self.search_documents(text_query, k=k) if text_query else []

        if not hasattr(self, 'multimodal_processor'):
            logger.error("Мультимодальный процессор недоступен")
            return self.search_documents(text_query, k=k) if text_query else []

        try:
            visual_query = None
            if image_query_path:
                visual_query = self.multimodal_processor.create_visual_embedding(Path(image_query_path))

            results = self.faiss_manager.search_multimodal(
                text_query=text_query,
                visual_query=visual_query,
                k=k,
                text_weight=text_weight
            )

            logger.info(
                f"🔍 Мультимодальный поиск: текст='{text_query}', изображение={'да' if image_query_path else 'нет'}, найдено={len(results)}")
            return results

        except Exception as e:
            logger.error(f"Ошибка мультимодального поиска: {e}")
            return []

    def get_image_analysis(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Анализ изображения с помощью CLIP

        Args:
            image_path: Путь к изображению

        Returns:
            Dict: Полный анализ изображения
        """
        if not self.enable_visual_search or not hasattr(self, 'multimodal_processor'):
            return {'error': 'Визуальный анализ недоступен'}

        try:
            image_path = Path(image_path)

            # Создаем визуальный вектор
            visual_vector = self.multimodal_processor.create_visual_embedding(image_path)

            # Определяем категории
            categories = self.multimodal_processor.get_image_categories(image_path)

            # Получаем топ-3 категории
            top_categories = dict(list(categories.items())[:3])

            # Определяем основной тип контента
            building_categories = ["фасад здания", "внешний вид здания", "архитектура"]
            document_categories = ["документ", "текст на бумаге", "чертеж", "схема"]

            building_scores = [categories.get(cat, 0) for cat in building_categories]
            document_scores = [categories.get(cat, 0) for cat in document_categories]

            max_building = max(building_scores) if building_scores else 0
            max_document = max(document_scores) if document_scores else 0

            main_type = "building" if max_building > max_document else "document"

            return {
                'image_path': str(image_path),
                'visual_vector_shape': visual_vector.shape,
                'main_content_type': main_type,
                'building_confidence': max_building,
                'document_confidence': max_document,
                'top_categories': top_categories,
                'all_categories': categories,
                'analysis_successful': True
            }

        except Exception as e:
            logger.error(f"Ошибка анализа изображения {image_path}: {e}")
            return {'error': str(e), 'analysis_successful': False}

    def find_images_by_category(self, category: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Находит все изображения определенной категории

        Args:
            category: Категория для поиска ("фасад", "документ", etc.)
            k: Максимальное количество результатов

        Returns:
            List[Dict]: Найденные изображения
        """
        # Поиск по текстовому описанию через визуальное пространство
        if self.enable_visual_search:
            return self.search_by_text_description(category, k=k, search_images_only=True)
        else:
            # Fallback на текстовый поиск с фильтром
            results = self.search_documents(category, k=k * 2)
            image_results = []
            for result in results:
                metadata = result.get('metadata', {})
                if metadata.get('is_image', False):
                    image_results.append(result)
            return image_results[:k]

    def get_similar_to_existing(self, source_file: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Находит изображения, похожие на уже существующее в индексе

        Args:
            source_file: Имя файла в индексе
            k: Количество похожих результатов

        Returns:
            List[Dict]: Похожие изображения
        """
        if not self.enable_visual_search:
            return []

        try:
            # Используем новый метод FAISS менеджера
            return self.faiss_manager.get_similar_visual_chunks(source_file, k=k)

        except Exception as e:
            logger.error(f"Ошибка поиска похожих на {source_file}: {e}")
            return []

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Применяет фильтры к результатам поиска"""
        filtered_results = []

        for result in results:
            metadata = result.get('metadata', {})
            should_include = True

            for filter_key, filter_value in filters.items():
                if filter_key not in metadata:
                    should_include = False
                    break

                if isinstance(filter_value, list):
                    if metadata[filter_key] not in filter_value:
                        should_include = False
                        break
                else:
                    if metadata[filter_key] != filter_value:
                        should_include = False
                        break

            if should_include:
                filtered_results.append(result)

        return filtered_results

    # Методы для обратной совместимости (не изменены)

    def get_document_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Получает все чанки определенного документа"""
        return self.faiss_manager.get_chunks_by_source(source_file)

    def remove_document(self, source_file: str) -> bool:
        """Удаляет документ из индекса"""
        chunks = self.faiss_manager.get_chunks_by_source(source_file)
        if chunks:
            chunk_ids = [chunk.get('chunk_id') for chunk in chunks if chunk.get('chunk_id')]
            success = self.faiss_manager.remove_chunks(chunk_ids)
            if success:
                self.faiss_manager.save_index()
                logger.info(f"Документ {source_file} удален из индекса")
            return success
        return False

    def update_document(self, json_file_path: str, source_file: str) -> Dict[str, Any]:
        """Обновляет конкретный документ"""
        # Сначала удаляем старую версию
        self.remove_document(source_file)

        # Затем обрабатываем заново
        return self.process_documents_from_json(json_file_path, update_existing=True)

    def get_index_statistics(self) -> Dict[str, Any]:
        """Возвращает расширенную статистику индекса"""
        base_stats = self.faiss_manager.get_index_statistics()

        # ✅ НОВАЯ статистика для мультимодального режима
        if base_stats.get('status') == 'ready':
            all_chunks = self.faiss_manager.get_all_chunks()

            # Анализ контента
            images_count = 0
            documents_count = 0
            multimodal_count = 0
            categories = {}
            file_types = {}
            total_chars = 0

            for chunk in all_chunks:
                metadata = chunk.get('metadata', {})

                # Подсчет типов
                if metadata.get('is_image', False):
                    images_count += 1
                else:
                    documents_count += 1

                if chunk.get('has_visual_vector', False):
                    multimodal_count += 1

                # Категории
                category = metadata.get('category', 'uncategorized')
                categories[category] = categories.get(category, 0) + 1

                # Типы файлов
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1

                # Общее количество символов
                total_chars += len(chunk.get('text', ''))

            # Добавляем новую статистику
            base_stats.update({
                'content_analysis': {
                    'images_count': images_count,
                    'documents_count': documents_count,
                    'multimodal_count': multimodal_count,
                    'visual_coverage': multimodal_count / len(all_chunks) if all_chunks else 0
                },
                'file_types_distribution': file_types,
                'categories_distribution': categories,
                'total_characters': total_chars,
                'average_chunk_size': total_chars / len(all_chunks) if all_chunks else 0,
                'search_capabilities': {
                    'text_search': True,
                    'visual_search': self.enable_visual_search,
                    'multimodal_search': self.enable_visual_search,
                    'similar_images': self.enable_visual_search,
                    'image_analysis': self.enable_visual_search
                }
            })

        return base_stats

    def clear_all_data(self, client_id: Optional[str] = None):
        """Очищает все данные (осторожно!)"""
        target_client = client_id or self.client_id
        logger.warning(f"Очищаем все данные из индекса клиента {target_client}")
        self.faiss_manager.clear_index()

    def export_chunks_to_json(self, output_path: str):
        """Экспортирует все чанки в JSON файл"""
        all_chunks = self.faiss_manager.get_all_chunks()

        export_data = {
            'exported_at': datetime.now().isoformat(),
            'client_id': self.client_id,
            'enable_visual_search': self.enable_visual_search,
            'total_chunks': len(all_chunks),
            'chunks': all_chunks
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Экспортировано {len(all_chunks)} чанков в {output_path}")

    # ✅ НОВЫЕ методы для мультимодального режима

    def get_processing_mode_info(self) -> Dict[str, Any]:
        """Возвращает информацию о режиме обработки"""
        mode_info = {
            'client_id': self.client_id,
            'visual_search_enabled': self.enable_visual_search,
            'capabilities': {
                'text_search': True,
                'image_search_by_description': self.enable_visual_search,
                'similar_image_search': self.enable_visual_search,
                'multimodal_combined_search': self.enable_visual_search,
                'image_analysis': self.enable_visual_search
            }
        }

        if self.enable_visual_search and hasattr(self, 'multimodal_processor'):
            mode_info['multimodal_model_info'] = self.multimodal_processor.get_model_info()

        return mode_info

    def export_visual_vectors(self, output_path: str):
        """Экспортирует визуальные векторы для анализа"""
        if not self.enable_visual_search:
            logger.warning("Нет визуальных векторов для экспорта")
            return

        try:
            export_data = self.faiss_manager.export_visual_vectors()

            if 'error' in export_data:
                logger.error(f"Ошибка экспорта: {export_data['error']}")
                return

            export_data['exported_at'] = datetime.now().isoformat()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Экспортировано {export_data['visual_vectors_count']} визуальных векторов в {output_path}")

        except Exception as e:
            logger.error(f"Ошибка экспорта визуальных векторов: {e}")


# ✅ НОВЫЕ функции для удобства создания процессоров

def create_text_processor(client_id: str) -> DocumentProcessor:
    """Создает процессор только для текстового поиска"""
    return DocumentProcessor(client_id=client_id, enable_visual_search=False)


def create_multimodal_processor(client_id: str) -> DocumentProcessor:
    """Создает процессор с поддержкой визуального поиска"""
    return DocumentProcessor(client_id=client_id, enable_visual_search=True)


def create_auto_processor(client_id: str) -> DocumentProcessor:
    """Создает процессор с автоопределением режима по существующей конфигурации"""
    return DocumentProcessor(client_id=client_id, enable_visual_search=None)


# Пример использования
def main():
    """Пример использования обновленного DocumentProcessor"""

    print("🚀 Пример использования обновленного DocumentProcessor")
    print("=" * 70)

    client_id = "demo_client"

    # Создаем процессор (автоопределение режима)
    processor = create_auto_processor(client_id)

    print("📊 Информация о процессоре:")
    mode_info = processor.get_processing_mode_info()
    for key, value in mode_info.items():
        if key != 'multimodal_model_info':
            print(f"   {key}: {value}")

    print(f"\n🔍 Доступные возможности поиска:")
    capabilities = mode_info['capabilities']
    for capability, available in capabilities.items():
        status = "✅" if available else "❌"
        print(f"   {status} {capability}")

    print(f"\n📝 Примеры использования:")

    print(f"\n1. Обычный текстовый поиск:")
    print(f"   results = processor.search_documents('фасад здания')")

    if processor.enable_visual_search:
        print(f"\n2. Поиск изображений по описанию:")
        print(f"   results = processor.search_by_text_description('современный фасад')")

        print(f"\n3. Поиск похожих изображений:")
        print(f"   results = processor.search_similar_images('my_facade.jpg')")

        print(f"\n4. Комбинированный поиск:")
        print(f"   results = processor.search_multimodal(")
        print(f"       text_query='фасад',")
        print(f"       image_query_path='example.jpg'")
        print(f"   )")

        print(f"\n5. Анализ изображения:")
        print(f"   analysis = processor.get_image_analysis('facade.jpg')")

    print(f"\n6. Статистика индекса:")
    print(f"   stats = processor.get_index_statistics()")

    print(f"\n✅ Обновленный DocumentProcessor готов к использованию!")


if __name__ == "__main__":
    main()