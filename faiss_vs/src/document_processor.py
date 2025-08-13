import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .data.loaders import DocumentLoader, DocumentParser
from .data.chunkers import DocumentChunker, TextChunk
from .data.image_processor import ImageProcessor
from .vectorstore.faiss_manager import FAISSManager
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Главный класс для обработки документов и управления RAG пайплайном"""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.document_loader = DocumentLoader(client_id=client_id)
        self.document_parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.image_processor = ImageProcessor()
        self.faiss_manager = FAISSManager(client_id=client_id)

        # Загружаем существующий индекс при инициализации
        self.faiss_manager.load_index()

    def process_documents_from_json(self, json_file_path: str,
                                    update_existing: bool = False) -> Dict[str, Any]:
        """
        Обрабатывает документы из JSON файла

        Args:
            json_file_path: Путь к JSON файлу со списком документов
            update_existing: Обновлять ли существующие документы

        Returns:
            Статистика обработки
        """
        logger.info(f"Начинаем обработку документов из {json_file_path}")

        # Статистика
        stats = {
            'total_documents': 0,
            'downloaded': 0,
            'parsed': 0,
            'chunked': 0,
            'indexed': 0,
            'errors': [],
            'processed_files': [],
            'start_time': datetime.now().isoformat()
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
                    original_metadata = doc_info['metadata']  # Это метаданные из JSON

                    logger.info(f"Обрабатываем: {file_path.name}")
                    logger.info(f"Исходные метаданные: {original_metadata}")

                    # Проверяем, нужно ли обновлять существующие чанки
                    if not update_existing:
                        existing_chunks = self.faiss_manager.get_chunks_by_source(file_path.name)
                        if existing_chunks:
                            logger.info(f"Документ {file_path.name} уже существует, пропускаем")
                            continue

                    stats['downloaded'] += 1

                    # Проверяем, это изображение или документ
                    if self.image_processor.is_image_file(file_path):
                        # Обрабатываем как изображение
                        text = self.image_processor.create_image_embedding_text(file_path, original_metadata)
                        logger.info(f"Обработано изображение: {file_path.name}")
                        doc_metadata = None  # Для изображений метаданные уже в тексте
                    else:
                        # Парсим как документ
                        text, doc_metadata = self.document_parser.parse_document(file_path)

                    if not text.strip():
                        stats['errors'].append(f"Пустой текст в {file_path.name}")
                        continue

                    stats['parsed'] += 1

                    # ПРАВИЛЬНО собираем все метаданные
                    enhanced_metadata = {
                        # Исходные метаданные из JSON (самые важные!)
                        'source_url': url,
                        'title': original_metadata.get('title', ''),
                        'description': original_metadata.get('description', ''),
                        'date': original_metadata.get('date', ''),
                        'guid_doc': original_metadata.get('guid', ''),
                        'parent': original_metadata.get('parent', ''),
                        'object_id': original_metadata.get('object_id', ''),
                        'category': original_metadata.get('category', original_metadata.get('parent', 'uncategorized')),

                        # Метаданные файла
                        'file_type': file_path.suffix.lower(),
                        'filename': file_path.name,
                        'file_size': file_path.stat().st_size if file_path.exists() else 0,

                        # Метаданные обработки
                        'processing_date': datetime.now().isoformat(),
                        'client_id': self.client_id,

                        # Метаданные документа (если есть)
                        **(doc_metadata.__dict__ if doc_metadata else {}),

                        # ВСЕ остальные поля из исходного JSON
                        **{k: v for k, v in original_metadata.items()
                           if k not in ['title', 'description', 'date', 'guid', 'parent', 'object_id', 'category']}
                    }

                    logger.info(f"Финальные метаданные для {file_path.name}: {enhanced_metadata}")

                    chunks = self.chunker.create_chunks(text, file_path.name, enhanced_metadata)

                    if chunks:
                        all_chunks.extend(chunks)
                        stats['chunked'] += len(chunks)
                        stats['processed_files'].append({
                            'filename': file_path.name,
                            'chunks_count': len(chunks),
                            'characters': len(text),
                            'url': url,
                            'metadata_keys': list(enhanced_metadata.keys())  # Для отладки
                        })
                        logger.info(f"Создано {len(chunks)} чанков для {file_path.name}")
                        logger.info(f"Метаданные первого чанка: {chunks[0].metadata}")
                        if not settings.KEEP_DOWNLOADED_FILES:
                            file_path.unlink()  # Удаляем файл
                            logger.info(f"Файл {file_path.name} удален после обработки")

                except Exception as e:
                    error_msg = f"Ошибка при обработке {doc_info.get('file_path', 'unknown')}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
                    continue

            # Добавляем все чанки в FAISS индекс
            if all_chunks:
                if update_existing:
                    # При обновлении сначала удаляем старые чанки тех же файлов
                    processed_files = [f['filename'] for f in stats['processed_files']]
                    for filename in processed_files:
                        existing_chunks = self.faiss_manager.get_chunks_by_source(filename)
                        if existing_chunks:
                            chunk_ids = [chunk['chunk_id'] for chunk in existing_chunks
                                         if 'chunk_id' in chunk]
                            self.faiss_manager.remove_chunks(chunk_ids)

                self.faiss_manager.add_chunks(all_chunks)
                stats['indexed'] = len(all_chunks)

                # Сохраняем индекс
                self.faiss_manager.save_index()
                logger.info(f"Индекс сохранен с {len(all_chunks)} новыми чанками")

            stats['end_time'] = datetime.now().isoformat()
            stats['success'] = True

            # Логируем итоговую статистику
            logger.info(f"Обработка завершена: {stats['indexed']} чанков добавлено в индекс")

        except Exception as e:
            stats['end_time'] = datetime.now().isoformat()
            stats['success'] = False
            stats['errors'].append(f"Критическая ошибка: {e}")
            logger.error(f"Критическая ошибка при обработке: {e}")

        return stats

    def search_documents(self, query: str, k: int = 5,
                         min_score: float = 0.0,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Поиск документов по запросу

        Args:
            query: Поисковый запрос
            k: Количество результатов
            min_score: Минимальный score для результатов
            filters: Фильтры по метаданным (например, {'category': 'tech'})

        Returns:
            Список найденных документов с релевантностью
        """
        results = self.faiss_manager.search(query, k=k * 2, score_threshold=min_score)

        # Применяем фильтры
        if filters:
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

            results = filtered_results

        # Ограничиваем количество результатов
        return results[:k]

    def get_document_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Получает все чанки определенного документа"""
        return self.faiss_manager.get_chunks_by_source(source_file)

    def remove_document(self, source_file: str) -> bool:
        """Удаляет документ из индекса"""
        chunks = self.faiss_manager.get_chunks_by_source(source_file)
        if chunks:
            chunk_ids = [chunk['chunk_id'] for chunk in chunks if 'chunk_id' in chunk]
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
        """Возвращает статистику индекса"""
        base_stats = self.faiss_manager.get_index_stats()

        # Добавляем дополнительную статистику
        if base_stats.get('status') == 'ready':
            all_chunks = self.faiss_manager.get_all_chunks()

            # Статистика по типам файлов
            file_types = {}
            categories = {}
            total_chars = 0

            for chunk in all_chunks:
                metadata = chunk.get('metadata', {})

                # Типы файлов
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1

                # Категории
                category = metadata.get('category', 'uncategorized')
                categories[category] = categories.get(category, 0) + 1

                # Общее количество символов
                total_chars += len(chunk.get('text', ''))

            base_stats.update({
                'file_types_distribution': file_types,
                'categories_distribution': categories,
                'total_characters': total_chars,
                'average_chunk_size': total_chars / len(all_chunks) if all_chunks else 0
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
            'total_chunks': len(all_chunks),
            'chunks': all_chunks
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Экспортировано {len(all_chunks)} чанков в {output_path}")


# Пример использования
def main():
    """Пример использования DocumentProcessor"""
    processor = DocumentProcessor(client_id="test_client")

    # Пример JSON файла с документами
    example_json = {
        "documents": [
            {
                "url": "https://example.com/document1.pdf",
                "title": "Техническая документация",
                "description": "Описание API",
                "category": "tech",
                "tags": ["api", "documentation"]
            },
            {
                "url": "https://example.com/document2.docx",
                "filename": "manual.docx",
                "title": "Руководство пользователя",
                "category": "manual"
            }
        ]
    }

    # Сохраняем пример
    with open('documents.json', 'w', encoding='utf-8') as f:
        json.dump(example_json, f, ensure_ascii=False, indent=2)

    print("Создан пример файла documents.json")
    print("Для обработки документов запустите:")
    print("stats = processor.process_documents_from_json('documents.json')")
    print("print(stats)")


if __name__ == "__main__":
    main()