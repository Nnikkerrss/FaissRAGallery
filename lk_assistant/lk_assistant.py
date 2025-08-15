#!/usr/bin/env python3
"""
Нейроассистент для личного кабинета на основе FAISS RAG
Файл: lk_assistant/lk_assistant.py
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Добавляем faiss_vs в путь для использования существующих компонентов
sys.path.append(str(Path(__file__).parent.parent / "faiss_vs"))

from faiss_vs.src.document_processor import DocumentProcessor
from faiss_vs.src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LKAssistant:
    """
    Нейроассистент для личного кабинета
    Использует FAISS RAG для поиска информации в документах клиента
    """

    def __init__(self, client_id: str, assistant_name: str = "Помощник ЛК"):
        self.client_id = client_id
        self.assistant_name = assistant_name
        self.conversation_history = []

        # Инициализируем процессор документов
        self.document_processor = DocumentProcessor(client_id=client_id)

        # Проверяем, есть ли данные для клиента
        self.is_ready = self._check_client_data()

        logger.info(f"Ассистент инициализирован для клиента {client_id}")

    def _check_client_data(self) -> bool:
        """Проверяет, есть ли индексированные данные для клиента"""
        try:
            stats = self.document_processor.get_index_statistics()
            if stats.get('status') == 'ready' and stats.get('total_chunks', 0) > 0:
                logger.info(f"Найдено {stats['total_chunks']} чанков для клиента {self.client_id}")
                return True
            else:
                logger.warning(f"Нет данных для клиента {self.client_id}")
                return False
        except Exception as e:
            logger.error(f"Ошибка проверки данных клиента: {e}")
            return False

    def ask(self, question: str, context_limit: int = 3) -> Dict[str, Any]:
        """
        Основной метод для вопросов к ассистенту

        Args:
            question: Вопрос пользователя
            context_limit: Максимальное количество документов для контекста

        Returns:
            Dict с ответом и метаданными
        """
        if not self.is_ready:
            return {
                'success': False,
                'answer': "Извините, для вашего аккаунта еще нет загруженных документов. Обратитесь к администратору.",
                'sources': [],
                'error': 'no_data'
            }

        try:
            # Ищем релевантные документы
            search_results = self.document_processor.search_documents(
                query=question,
                k=context_limit,
                min_score=0.1  # Минимальный порог релевантности
            )

            if not search_results:
                return {
                    'success': True,
                    'answer': f"Я не нашел информации по вашему вопросу в загруженных документах. Попробуйте переформулировать вопрос или обратиться к специалисту.",
                    'sources': [],
                    'query': question
                }

            # Формируем ответ на основе найденных документов
            answer = self._generate_answer(question, search_results)

            # Сохраняем в историю
            self._add_to_history(question, answer, search_results)

            return {
                'success': True,
                'answer': answer,
                'sources': self._format_sources(search_results),
                'query': question,
                'found_documents': len(search_results)
            }

        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса: {e}")
            return {
                'success': False,
                'answer': "Произошла ошибка при поиске ответа. Попробуйте еще раз.",
                'sources': [],
                'error': str(e)
            }

    def _generate_answer(self, question: str, search_results: List[Dict]) -> str:
        """
        Генерирует ответ на основе найденных документов
        Пока простая реализация, потом можно добавить LLM
        """
        if not search_results:
            return "Информация не найдена."

        # Извлекаем самые релевантные фрагменты
        relevant_texts = []
        for result in search_results:
            text = result.get('text', '')
            score = result.get('score', 0)
            source = result.get('source_file', 'Неизвестный источник')

            # Добавляем фрагмент с указанием источника
            relevant_texts.append(f"[Из документа '{source}'] {text[:300]}...")

        # Формируем ответ
        answer_parts = [
            f"На основе найденной информации в ваших документах:",
            "",
        ]

        # Добавляем релевантные фрагменты
        for i, text in enumerate(relevant_texts, 1):
            answer_parts.append(f"{i}. {text}")
            answer_parts.append("")

        # Добавляем примечание
        answer_parts.append(
            "📝 Это информация из ваших загруженных документов. Если нужны уточнения, задайте более конкретный вопрос.")

        return "\n".join(answer_parts)

    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        """Форматирует источники для ответа"""
        sources = []
        for result in search_results:
            metadata = result.get('metadata', {})
            source = {
                'file': result.get('source_file', 'Неизвестный файл'),
                'score': round(result.get('score', 0), 3),
                'category': metadata.get('category', 'Без категории'),
                'title': metadata.get('title', ''),
                'url': metadata.get('source_url', ''),
                'date': metadata.get('date', '')
            }
            sources.append(source)
        return sources

    def _add_to_history(self, question: str, answer: str, sources: List[Dict]):
        """Добавляет взаимодействие в историю"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'sources_count': len(sources),
            'client_id': self.client_id
        })

    def get_client_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по данным клиента"""
        if not self.is_ready:
            return {'error': 'Нет данных для клиента'}

        stats = self.document_processor.get_index_statistics()

        return {
            'client_id': self.client_id,
            'total_documents': stats.get('sources_count', 0),
            'total_chunks': stats.get('total_chunks', 0),
            'categories': list(stats.get('categories_distribution', {}).keys()),
            'conversation_history_length': len(self.conversation_history),
            'is_ready': self.is_ready
        }

    def search_by_category(self, query: str, category: str) -> Dict[str, Any]:
        """Поиск только в определенной категории"""
        if not self.is_ready:
            return {'error': 'Нет данных для клиента'}

        try:
            # Используем фильтры для поиска в конкретной категории
            results = self.document_processor.search_documents(
                query=query,
                k=5,
                filters={'category': category}
            )

            return {
                'success': True,
                'results': results,
                'category': category,
                'found': len(results)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Возвращает историю разговора"""
        return self.conversation_history[-limit:] if limit else self.conversation_history

    def clear_history(self):
        """Очищает историю разговора"""
        self.conversation_history = []
        logger.info(f"История разговора очищена для клиента {self.client_id}")

    def suggest_questions(self) -> List[str]:
        """Предлагает вопросы на основе доступных данных"""
        if not self.is_ready:
            return ["Сначала загрузите документы в систему"]

        stats = self.document_processor.get_index_statistics()
        categories = list(stats.get('categories_distribution', {}).keys())

        suggestions = [
            "Что содержится в моих документах?",
            "Покажи последние загруженные файлы",
        ]

        # Добавляем вопросы по категориям
        for category in categories[:3]:  # Берем первые 3 категории
            suggestions.append(f"Что есть в категории '{category}'?")

        suggestions.extend([
            "Найди информацию о проекте",
            "Покажи техническую документацию",
            "Есть ли информация о ценах?"
        ])

        return suggestions

    def get_available_categories(self) -> List[str]:
        """Возвращает список доступных категорий документов"""
        if not self.is_ready:
            return []

        stats = self.document_processor.get_index_statistics()
        return list(stats.get('categories_distribution', {}).keys())

    def get_recent_documents(self, limit: int = 5) -> List[Dict]:
        """Возвращает список недавно добавленных документов"""
        if not self.is_ready:
            return []

        try:
            all_chunks = self.document_processor.faiss_manager.get_all_chunks()

            # Группируем по файлам и берем последние
            files_info = {}
            for chunk in all_chunks:
                filename = chunk.get('source_file', '')
                if filename not in files_info:
                    metadata = chunk.get('metadata', {})
                    files_info[filename] = {
                        'filename': filename,
                        'category': metadata.get('category', 'Без категории'),
                        'title': metadata.get('title', ''),
                        'date': metadata.get('processing_date', ''),
                        'chunks_count': 0
                    }
                files_info[filename]['chunks_count'] += 1

            # Сортируем по дате и возвращаем последние
            recent_files = sorted(
                files_info.values(),
                key=lambda x: x.get('date', ''),
                reverse=True
            )

            return recent_files[:limit]

        except Exception as e:
            logger.error(f"Ошибка получения недавних документов: {e}")
            return []