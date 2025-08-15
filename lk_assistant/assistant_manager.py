#!/usr/bin/env python3
"""
Менеджер для управления ассистентами личного кабинета
Файл: lk_assistant/assistant_manager.py
"""

import logging
import threading
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from .lk_assistant import LKAssistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssistantManager:
    """
    Менеджер для управления несколькими ассистентами
    Обеспечивает кэширование, автоочистку и мониторинг
    """

    def __init__(self, cache_ttl_minutes: int = 60, max_assistants: int = 100):
        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_assistants = max_assistants
        self.assistants_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        logger.info(f"AssistantManager инициализирован: TTL={cache_ttl_minutes}мин, MAX={max_assistants}")

    def get_assistant(self, client_id: str, force_reload: bool = False) -> LKAssistant:
        """
        Получает ассистента для клиента (создает или возвращает из кэша)

        Args:
            client_id: ID клиента
            force_reload: Принудительно пересоздать ассистента

        Returns:
            LKAssistant: Экземпляр ассистента
        """
        with self.lock:
            # Очищаем устаревшие записи
            self._cleanup_expired()

            # Проверяем лимит
            if len(self.assistants_cache) >= self.max_assistants:
                self._cleanup_oldest()

            # Принудительная перезагрузка
            if force_reload and client_id in self.assistants_cache:
                logger.info(f"Принудительно перезагружаем ассистента для клиента {client_id}")
                del self.assistants_cache[client_id]

            # Проверяем кэш
            if client_id in self.assistants_cache:
                cache_entry = self.assistants_cache[client_id]

                # Проверяем TTL
                if self._is_cache_valid(cache_entry):
                    cache_entry['last_accessed'] = datetime.now()
                    cache_entry['access_count'] += 1
                    logger.debug(f"Возвращаем ассистента из кэша для клиента {client_id}")
                    return cache_entry['assistant']
                else:
                    # Удаляем устаревшую запись
                    del self.assistants_cache[client_id]

            # Создаем нового ассистента
            logger.info(f"Создаем нового ассистента для клиента {client_id}")
            assistant = LKAssistant(client_id=client_id)

            # Добавляем в кэш
            self.assistants_cache[client_id] = {
                'assistant': assistant,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'access_count': 1
            }

            return assistant

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Проверяет валидность записи в кэше"""
        expiry_time = cache_entry['last_accessed'] + timedelta(minutes=self.cache_ttl_minutes)
        return datetime.now() < expiry_time

    def _cleanup_expired(self):
        """Удаляет устаревшие записи из кэша"""
        expired_clients = []

        for client_id, cache_entry in self.assistants_cache.items():
            if not self._is_cache_valid(cache_entry):
                expired_clients.append(client_id)

        for client_id in expired_clients:
            logger.info(f"Удаляем устаревшего ассистента для клиента {client_id}")
            del self.assistants_cache[client_id]

    def _cleanup_oldest(self):
        """Удаляет самую старую запись для освобождения места"""
        if not self.assistants_cache:
            return

        oldest_client = min(
            self.assistants_cache.keys(),
            key=lambda client_id: self.assistants_cache[client_id]['last_accessed']
        )

        logger.info(f"Удаляем самого старого ассистента для клиента {oldest_client}")
        del self.assistants_cache[oldest_client]

    def remove_assistant(self, client_id: str) -> bool:
        """
        Удаляет ассистента из кэша

        Args:
            client_id: ID клиента

        Returns:
            bool: True если ассистент был удален
        """
        with self.lock:
            if client_id in self.assistants_cache:
                del self.assistants_cache[client_id]
                logger.info(f"Ассистент для клиента {client_id} удален из кэша")
                return True
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        with self.lock:
            total_assistants = len(self.assistants_cache)

            if total_assistants == 0:
                return {
                    'total_assistants': 0,
                    'cache_utilization': 0.0,
                    'oldest_entry_age_minutes': 0,
                    'most_accessed_client': None
                }

            # Находим самую старую запись
            oldest_time = min(
                entry['created_at'] for entry in self.assistants_cache.values()
            )
            oldest_age_minutes = (datetime.now() - oldest_time).total_seconds() / 60

            # Находим самого активного клиента
            most_accessed_client = max(
                self.assistants_cache.items(),
                key=lambda x: x[1]['access_count']
            )

            return {
                'total_assistants': total_assistants,
                'cache_utilization': (total_assistants / self.max_assistants) * 100,
                'oldest_entry_age_minutes': round(oldest_age_minutes, 2),
                'most_accessed_client': {
                    'client_id': most_accessed_client[0],
                    'access_count': most_accessed_client[1]['access_count']
                },
                'max_assistants': self.max_assistants,
                'cache_ttl_minutes': self.cache_ttl_minutes
            }

    def get_assistant_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию об ассистенте без его создания

        Args:
            client_id: ID клиента

        Returns:
            Dict или None если ассистент не найден в кэше
        """
        with self.lock:
            if client_id not in self.assistants_cache:
                return None

            cache_entry = self.assistants_cache[client_id]
            assistant = cache_entry['assistant']

            return {
                'client_id': client_id,
                'is_ready': assistant.is_ready,
                'assistant_name': assistant.assistant_name,
                'conversation_length': len(assistant.conversation_history),
                'cached_since': cache_entry['created_at'].isoformat(),
                'last_accessed': cache_entry['last_accessed'].isoformat(),
                'access_count': cache_entry['access_count'],
                'cache_valid': self._is_cache_valid(cache_entry)
            }

    def list_active_assistants(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает список всех активных ассистентов"""
        with self.lock:
            active_assistants = {}

            for client_id in self.assistants_cache.keys():
                info = self.get_assistant_info(client_id)
                if info:
                    active_assistants[client_id] = info

            return active_assistants

    def clear_all_cache(self) -> int:
        """
        Очищает весь кэш ассистентов

        Returns:
            int: Количество удаленных ассистентов
        """
        with self.lock:
            count = len(self.assistants_cache)
            self.assistants_cache.clear()
            logger.info(f"Очищен кэш, удалено {count} ассистентов")
            return count

    def update_assistant_settings(self,
                                  cache_ttl_minutes: Optional[int] = None,
                                  max_assistants: Optional[int] = None):
        """
        Обновляет настройки менеджера

        Args:
            cache_ttl_minutes: Новое время жизни кэша в минутах
            max_assistants: Новый максимум ассистентов
        """
        with self.lock:
            if cache_ttl_minutes is not None:
                self.cache_ttl_minutes = cache_ttl_minutes
                logger.info(f"Cache TTL обновлен до {cache_ttl_minutes} минут")

            if max_assistants is not None:
                self.max_assistants = max_assistants
                logger.info(f"Max assistants обновлен до {max_assistants}")

                # Если новый лимит меньше текущего количества, очищаем лишние
                while len(self.assistants_cache) > max_assistants:
                    self._cleanup_oldest()


# Глобальный экземпляр менеджера (Singleton)
_assistant_manager_instance = None


def get_assistant_manager() -> AssistantManager:
    """
    Возвращает глобальный экземпляр менеджера ассистентов

    Returns:
        AssistantManager: Экземпляр менеджера
    """
    global _assistant_manager_instance

    if _assistant_manager_instance is None:
        _assistant_manager_instance = AssistantManager()

    return _assistant_manager_instance