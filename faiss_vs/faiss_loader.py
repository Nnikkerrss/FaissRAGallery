#!/usr/bin/env python3
"""
Продакшн модуль для загрузки документов в FAISS из 1С API
"""

import json
import sys
from pathlib import Path
import requests
from urllib.parse import urlparse, parse_qs
import logging
from typing import Optional, Dict, Any
# Добавляем src в путь
sys.path.append(str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.config import settings


class DocumentLoader:
    """Класс для загрузки и обработки документов в FAISS"""

    def __init__(self, log_level: str = "INFO"):
        """Инициализация загрузчика"""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, log_level: str):
        """Настройка логирования"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def download_json_data(self, url: str) -> tuple[Dict[Any, Any], str]:
        """
        Скачивает JSON данные с указанного URL

        Args:
            url: URL для загрузки JSON данных

        Returns:
            Tuple[Dict, str]: (данные JSON, client_id)

        Raises:
            requests.RequestException: При ошибке загрузки
            ValueError: При ошибке парсинга client_id
        """
        self.logger.info(f"Загружаем данные с: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Извлекаем client_id из URL
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        client_id = query_params.get('client_id', [None])[0]

        if not client_id:
            raise ValueError("Не удалось извлечь client_id из URL")

        self.logger.info(f"Client ID: {client_id}")
        return data, client_id

    def save_json_data(self, data: Dict[Any, Any], client_id: str) -> str:
        """
        Сохраняет JSON данные в файл

        Args:
            data: JSON данные для сохранения
            client_id: ID клиента

        Returns:
            str: Путь к сохраненному файлу
        """
        # folder_path = Path("faiss") / "clients" / client_id
        folder_path = settings.FAISS_INDEX_DIR / "clients" / client_id
        folder_path.mkdir(parents=True, exist_ok=True)

        filename = folder_path / "data.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Данные сохранены: {filename}")
        return str(filename)

    def filter_processable_documents(self, documents: list) -> list:
        """
        Фильтрует документы, оставляя только обрабатываемые (PDF)

        Args:
            documents: Список документов из JSON

        Returns:
            list: Отфильтрованные PDF документы
        """
        pdf_docs = [doc for doc in documents if doc['ID'].lower().endswith('.pdf')]

        self.logger.info(f"Всего документов: {len(documents)}, PDF для обработки: {len(pdf_docs)}")
        return pdf_docs

    def process_documents(self, json_url: str) -> Dict[str, Any]:
        """
        Основная функция для обработки документов

        Args:
            json_url: URL для загрузки JSON с данными о документах

        Returns:
            Dict: Статистика обработки

        Raises:
            Exception: При любых ошибках обработки
        """
        try:
            # Загружаем JSON данные
            data, client_id = self.download_json_data(json_url)

            # Сохраняем JSON файл
            json_path = self.save_json_data(data, client_id)

            # Фильтруем документы
            documents = data.get('result', [])
            pdf_docs = self.filter_processable_documents(documents)

            if not pdf_docs:
                self.logger.warning("Нет PDF документов для обработки")
                return {
                    'success': False,
                    'error': 'Нет PDF документов для обработки',
                    'client_id': client_id,
                    'total_documents': len(documents),
                    'pdf_documents': 0
                }

            # Создаем процессор документов
            processor = DocumentProcessor(client_id=client_id)

            # Обрабатываем документы
            self.logger.info(f"Начинаем обработку {len(pdf_docs)} PDF документов...")
            stats = processor.process_documents_from_json(json_path)

            # Дополняем статистику
            stats['success'] = True
            stats['client_id'] = client_id
            stats['json_path'] = json_path

            self.logger.info(f"Обработка завершена. Успешно проиндексировано: {stats['indexed']} документов")

            return stats

        except Exception as e:
            error_msg = f"Ошибка при обработке документов: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def get_processing_status(self, client_id: str) -> Dict[str, Any]:
        """
        Получает статус обработки документов клиента

        Args:
            client_id: ID клиента

        Returns:
            Dict: Информация о статусе
        """
        try:
            processor = DocumentProcessor(client_id=client_id)
            stats = processor.get_index_statistics()

            return {
                'success': True,
                'client_id': client_id,
                'index_exists': stats.get('total_vectors', 0) > 0,
                'statistics': stats
            }

        except Exception as e:
            return {
                'success': False,
                'client_id': client_id,
                'error': str(e)
            }


def load_documents_from_url(json_url: str, log_level: str = "INFO") -> Dict[str, Any]:
    """
    Функция для загрузки документов из 1С API

    Args:
        json_url: URL API 1С с данными о документах
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Dict: Результат обработки со статистикой

    Example:
        >>> result = load_documents_from_url(
        ...     "https://1c.gwd.ru/services/hs/Request/GetData/GetAllFiles"
        ...     "?api_key=xxx&client_id=xxx"
        ... )
        >>> print(f"Обработано: {result['indexed']} документов")
    """
    loader = DocumentLoader(log_level=log_level)
    return loader.process_documents(json_url)


def get_client_status(client_id: str) -> Dict[str, Any]:
    """
    Получает статус обработки клиента

    Args:
        client_id: ID клиента

    Returns:
        Dict: Статус и статистика клиента
    """
    loader = DocumentLoader(log_level="WARNING")  # Минимальное логирование
    return loader.get_processing_status(client_id)


if __name__ == "__main__":
    """
    Использование из командной строки:
    python document_loader.py <json_url>
    """
    if len(sys.argv) != 2:
        print("Использование: python document_loader.py <json_url>")
        sys.exit(1)

    json_url = sys.argv[1]

    try:
        result = load_documents_from_url(json_url)

        if result['success']:
            print(f"✅ Успешно обработано {result['indexed']} документов")
            print(f"Client ID: {result['client_id']}")
            print(f"Всего документов: {result['total_documents']}")
            print(f"Скачано: {result['downloaded']}")
            print(f"Проанализировано: {result['parsed']}")
            print(f"Создано чанков: {result['chunked']}")
        else:
            print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)