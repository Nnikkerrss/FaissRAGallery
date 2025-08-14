#!/usr/bin/env python3
"""
Модуль для улучшения релевантности поиска в RAG системе
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchConfig:
    """Конфигурация поиска"""
    min_score_threshold: float = 0.3  # Минимальный score для показа
    category_boost: float = 1.2  # Усиление поиска по категориям
    title_boost: float = 1.3  # Усиление поиска по заголовкам
    description_boost: float = 1.1  # Усиление поиска по описаниям
    exact_match_boost: float = 1.5  # Усиление точных совпадений
    semantic_weight: float = 0.7  # Вес семантического поиска
    keyword_weight: float = 0.3  # Вес ключевых слов


class RelevanceImprover:
    """Класс для улучшения релевантности поиска"""

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()

        # Словари синонимов для строительной тематики
        self.building_synonyms = {
            'окна': ['окно', 'оконный', 'остекление', 'рама', 'стеклопакет'],
            'двери': ['дверь', 'дверной', 'проем', 'полотно', 'коробка'],
            'спецификация': ['перечень', 'список', 'состав', 'ведомость', 'таблица'],
            'фасад': ['фасадный', 'наружный', 'внешний', 'облицовка'],
            'геология': ['геологический', 'грунт', 'изыскания', 'почва'],
            'проект': ['проектный', 'чертеж', 'план', 'схема']
        }

        # Категории документов
        self.document_categories = {
            'specifications': ['спецификация', 'ведомость', 'перечень', 'таблица'],
            'geology': ['геология', 'изыскания', 'грунт', 'почва'],
            'photos': ['фото', 'изображение', 'снимок', 'img'],
            'drawings': ['чертеж', 'план', 'схема', 'проект']
        }

    def expand_query(self, query: str) -> List[str]:
        """Расширяет поисковый запрос синонимами"""
        expanded_queries = [query]
        query_lower = query.lower()

        for base_word, synonyms in self.building_synonyms.items():
            if base_word in query_lower:
                for synonym in synonyms:
                    new_query = query_lower.replace(base_word, synonym)
                    if new_query != query_lower:
                        expanded_queries.append(new_query)

        return expanded_queries

    def calculate_keyword_relevance(self, query: str, result: Dict[str, Any]) -> float:
        """Вычисляет релевантность на основе ключевых слов"""
        query_words = set(query.lower().split())
        score = 0.0

        # Проверяем разные поля результата
        fields_to_check = {
            'text': 1.0,
            'metadata.description': self.config.description_boost,
            'metadata.title': self.config.title_boost,
            'metadata.category': self.config.category_boost,
            'source_file': 0.8
        }

        for field_path, weight in fields_to_check.items():
            field_value = self._get_nested_field(result, field_path)
            if field_value:
                field_words = set(str(field_value).lower().split())
                matches = len(query_words.intersection(field_words))
                if matches > 0:
                    score += (matches / len(query_words)) * weight

        return min(score, 2.0)  # Ограничиваем максимальный score

    def _get_nested_field(self, obj: Dict, field_path: str) -> Any:
        """Получает значение вложенного поля"""
        try:
            keys = field_path.split('.')
            value = obj
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None

    def detect_query_intent(self, query: str) -> Dict[str, float]:
        """Определяет намерение поискового запроса"""
        intent_scores = {
            'specifications': 0.0,
            'geology': 0.0,
            'photos': 0.0,
            'drawings': 0.0,
            'general': 0.5
        }

        query_lower = query.lower()

        for category, keywords in self.document_categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent_scores[category] += 1.0

        # Нормализуем scores
        max_score = max(intent_scores.values())
        if max_score > 0:
            for category in intent_scores:
                intent_scores[category] /= max_score

        return intent_scores

    def filter_by_intent(self, results: List[Dict[str, Any]], query_intent: Dict[str, float]) -> List[Dict[str, Any]]:
        """Фильтрует результаты по намерению запроса"""
        filtered_results = []

        for result in results:
            metadata = result.get('metadata', {})
            category = metadata.get('category', '').lower()
            filename = result.get('source_file', '').lower()

            # Определяем тип документа
            doc_type = 'general'

            if any(keyword in category or keyword in filename
                   for keyword in self.document_categories['specifications']):
                doc_type = 'specifications'
            elif any(keyword in category or keyword in filename
                     for keyword in self.document_categories['geology']):
                doc_type = 'geology'
            elif filename.endswith(('.jpg', '.png', '.jpeg')):
                doc_type = 'photos'
            elif any(keyword in category or keyword in filename
                     for keyword in self.document_categories['drawings']):
                doc_type = 'drawings'

            # Применяем фильтр намерения
            intent_match = query_intent.get(doc_type, 0.0)

            # Если намерение очень специфично, фильтруем строго
            max_intent = max(query_intent.values())
            if max_intent > 0.8:  # Очень специфичный запрос
                if intent_match < 0.5:
                    continue  # Пропускаем нерелевантные документы

            # Добавляем score намерения к результату
            result['intent_score'] = intent_match
            filtered_results.append(result)

        return filtered_results

    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Перенранжирует результаты для улучшения релевантности"""

        # 1. Определяем намерение запроса
        query_intent = self.detect_query_intent(query)

        # 2. Фильтруем по намерению
        filtered_results = self.filter_by_intent(results, query_intent)

        # 3. Пересчитываем scores
        for result in filtered_results:
            original_score = result.get('score', 0.0)
            keyword_score = self.calculate_keyword_relevance(query, result)
            intent_score = result.get('intent_score', 0.0)

            # Комбинированный score
            combined_score = (
                    original_score * self.config.semantic_weight +
                    keyword_score * self.config.keyword_weight +
                    intent_score * 0.3
            )

            result['original_score'] = original_score
            result['keyword_score'] = keyword_score
            result['intent_score'] = intent_score
            result['combined_score'] = combined_score
            result['score'] = combined_score  # Обновляем основной score

        # 4. Сортируем по новому score
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)

        # 5. Применяем минимальный порог
        final_results = [
            result for result in filtered_results
            if result['combined_score'] >= self.config.min_score_threshold
        ]

        return final_results


class SmartSearchEngine:
    """Умный поисковик с улучшенной релевантностью"""

    def __init__(self, document_processor, config: SearchConfig = None):
        self.processor = document_processor
        self.relevance_improver = RelevanceImprover(config)

    def smart_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Умный поиск с улучшенной релевантностью"""

        # 1. Расширяем запрос синонимами
        expanded_queries = self.relevance_improver.expand_query(query)

        # 2. Выполняем поиск по всем вариантам
        all_results = []
        seen_chunks = set()

        for expanded_query in expanded_queries:
            results = self.processor.search_documents(expanded_query, k=k * 2)

            for result in results:
                chunk_id = result.get('chunk_id')
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(result)

        # 3. Улучшаем релевантность
        improved_results = self.relevance_improver.rerank_results(query, all_results)

        # 4. Ограничиваем количество
        return improved_results[:k]