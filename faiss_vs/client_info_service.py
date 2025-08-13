"""
Сервис для получения информации о клиенте и его индексах
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

from src.document_processor import DocumentProcessor
from src.config import settings


class ClientInfoService:
    """Сервис для работы с информацией о клиентах"""

    def __init__(self):
        self.settings = settings

    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """
        Получает полную информацию о клиенте по его ID

        Args:
            client_id: ID клиента

        Returns:
            Dict: Полная информация о клиенте и его данных
        """
        try:
            # Путь к папке клиента
            client_folder = self.settings.FAISS_INDEX_DIR / "clients" / client_id

            if not client_folder.exists():
                return {
                    'success': False,
                    'error': f'Клиент {client_id} не найден',
                    'client_id': client_id
                }

            # Создаем процессор для получения статистики индекса
            processor = DocumentProcessor(client_id=client_id)

            # Собираем всю информацию
            result = {
                'success': True,
                'client_id': client_id,
                'client_folder': str(client_folder),
                'index_statistics': self._get_index_statistics(processor),
                'files': self._get_files_info(client_folder),
                'config': self._get_config_data(client_folder),
                'mappings': self._get_mappings_data(client_folder),
                'metadata': self._get_metadata_info(client_folder),
                'original_data': self._get_original_data_info(client_folder)
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка при получении данных: {str(e)}',
                'client_id': client_id
            }

    def _get_index_statistics(self, processor: DocumentProcessor) -> Dict[str, Any]:
        """Получает статистику индекса"""
        try:
            return processor.get_index_statistics()
        except Exception as e:
            return {'error': f'Ошибка получения статистики: {str(e)}'}

    def _get_files_info(self, client_folder: Path) -> Dict[str, Dict[str, Any]]:
        """Получает информацию о файлах в папке клиента"""
        files_info = {}
        try:
            for file_path in client_folder.glob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    files_info[file_path.name] = {
                        'size': file_size,
                        'size_mb': round(file_size / 1024 / 1024, 2),
                        'exists': True
                    }
        except Exception as e:
            files_info['error'] = f'Ошибка чтения файлов: {str(e)}'

        return files_info

    def _get_config_data(self, client_folder: Path) -> Dict[str, Any]:
        """Загружает конфигурацию клиента"""
        config_file = client_folder / "config.json"
        if not config_file.exists():
            return {'exists': False}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                config_data['exists'] = True
                return config_data
        except Exception as e:
            return {'error': f'Ошибка чтения config.json: {str(e)}', 'exists': True}

    def _get_mappings_data(self, client_folder: Path) -> Dict[str, Any]:
        """Загружает данные маппингов"""
        mappings_file = client_folder / "mappings.json"
        if not mappings_file.exists():
            return {'exists': False}

        try:
            with open(mappings_file, 'r', encoding='utf-8') as f:
                mappings_raw = json.load(f)
                return {
                    'exists': True,
                    'id_to_chunk_count': len(mappings_raw.get('id_to_chunk_id', {})),
                    'chunk_to_id_count': len(mappings_raw.get('chunk_id_to_id', {})),
                    'sample_mappings': dict(list(mappings_raw.get('id_to_chunk_id', {}).items())[:3])
                }
        except Exception as e:
            return {'error': f'Ошибка чтения mappings.json: {str(e)}', 'exists': True}

    def _get_metadata_info(self, client_folder: Path) -> Dict[str, Any]:
        """Загружает информацию о метаданных"""
        metadata_file = client_folder / "metadata.pkl"
        if not metadata_file.exists():
            return {'exists': False}

        try:
            with open(metadata_file, 'rb') as f:
                metadata_raw = pickle.load(f)

                if isinstance(metadata_raw, (list, tuple)):
                    result = {
                        'exists': True,
                        'chunks_count': len(metadata_raw),
                        'type': 'list'
                    }

                    # Добавляем пример первого чанка (безопасно)
                    if len(metadata_raw) > 0:
                        sample_chunk = metadata_raw[0]
                        if isinstance(sample_chunk, dict):
                            # Показываем только ключи и типы значений
                            result['sample_chunk_keys'] = list(sample_chunk.keys())
                            result['sample_chunk_types'] = {k: str(type(v).__name__) for k, v in sample_chunk.items()}
                        else:
                            result['sample_chunk_type'] = str(type(sample_chunk).__name__)

                    return result
                else:
                    return {
                        'exists': True,
                        'type': str(type(metadata_raw).__name__),
                        'data_preview': str(metadata_raw)[:200] + "..." if len(str(metadata_raw)) > 200 else str(
                            metadata_raw)
                    }
        except Exception as e:
            return {'error': f'Ошибка чтения metadata.pkl: {str(e)}', 'exists': True}

    def _get_original_data_info(self, client_folder: Path) -> Dict[str, Any]:
        """Загружает информацию об исходных данных"""
        data_file = client_folder / "data.json"
        if not data_file.exists():
            return {'exists': False}

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

                result = {
                    'exists': True,
                    'has_result': 'result' in original_data,
                    'documents_count': len(original_data.get('result', []))
                }

                # Добавляем пример документа (только ключи)
                documents = original_data.get('result', [])
                if documents:
                    sample_doc = documents[0]
                    if isinstance(sample_doc, dict):
                        result['sample_document_keys'] = list(sample_doc.keys())

                return result
        except Exception as e:
            return {'error': f'Ошибка чтения data.json: {str(e)}', 'exists': True}

    def client_exists(self, client_id: str) -> bool:
        """Проверяет существование клиента"""
        client_folder = self.settings.FAISS_INDEX_DIR / "clients" / client_id
        return client_folder.exists()

    def get_all_clients(self) -> Dict[str, Any]:
        """Получает список всех клиентов"""
        try:
            clients_dir = self.settings.FAISS_INDEX_DIR / "clients"
            if not clients_dir.exists():
                return {'success': True, 'clients': [], 'count': 0}

            clients = []
            for client_folder in clients_dir.iterdir():
                if client_folder.is_dir():
                    client_id = client_folder.name

                    # Получаем базовую информацию
                    files_count = len(list(client_folder.glob("*")))
                    has_index = (client_folder / "index.faiss").exists()

                    clients.append({
                        'client_id': client_id,
                        'folder_name': client_folder.name,
                        'files_count': files_count,
                        'has_index': has_index
                    })

            return {
                'success': True,
                'clients': clients,
                'count': len(clients)
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка получения списка клиентов: {str(e)}'
            }