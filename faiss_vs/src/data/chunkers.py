import re
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import hashlib

from ..config import settings


@dataclass
class TextChunk:
    text: str
    chunk_id: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]
    start_char: int = 0
    end_char: int = 0


class DocumentChunker:
    """Класс для разбивки документов на фрагменты"""

    def __init__(self,
                 chunk_size: int = settings.CHUNK_SIZE,
                 chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Инициализируем text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def preprocess_text(self, text: str) -> str:
        """Предобработка текста перед чанкингом"""
        # Удаляем лишние пробелы и переносы
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Двойные переносы оставляем
        text = re.sub(r'[ \t]+', ' ', text)  # Множественные пробелы в один
        text = text.strip()

        # Убираем очень короткие строки (возможно, артефакты)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3 or line == "":  # Оставляем пустые строки как разделители
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def create_chunks(self, text: str, source_file: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Создает чанки из текста"""
        if metadata is None:
            metadata = {}

        # Предобработка
        preprocessed_text = self.preprocess_text(text)

        if not preprocessed_text.strip():
            return []

        # Разбиваем на чанки
        chunks_text = self.text_splitter.split_text(preprocessed_text)

        chunks = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip():
                continue

            # Находим позицию чанка в оригинальном тексте
            start_pos = preprocessed_text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos - self.chunk_overlap  # Учитываем overlap

            # Создаем уникальный ID для чанка
            chunk_content = f"{source_file}_{i}_{chunk_text[:100]}"
            chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()

            # ИСПРАВЛЕНО: Правильно копируем ВСЕ метаданные в чанк
            chunk_metadata = {}

            # Сначала копируем ВСЕ исходные метаданные
            for key, value in metadata.items():
                chunk_metadata[key] = value

            # Затем добавляем метаданные чанка (не перезаписывая исходные!)
            chunk_specific_metadata = {
                'chunk_size': len(chunk_text),
                'chunk_tokens_estimate': len(chunk_text.split()),
                'chunk_index': i,
                'chunk_id': chunk_id,
                'start_char': start_pos,
                'end_char': end_pos,
                'has_tables': 'table' in chunk_text.lower() or '|' in chunk_text,
                'has_lists': bool(re.search(r'^\s*[\-\*\d+]\s', chunk_text, re.MULTILINE)),
            }

            # Добавляем чанк-специфичные метаданные
            chunk_metadata.update(chunk_specific_metadata)

            print(f"Чанк {i} для {source_file}: метаданных={len(chunk_metadata)}, ключи={list(chunk_metadata.keys())}")

            chunk = TextChunk(
                text=chunk_text.strip(),
                chunk_id=chunk_id,
                source_file=source_file,
                chunk_index=i,
                metadata=chunk_metadata,
                start_char=start_pos,
                end_char=end_pos
            )

            chunks.append(chunk)

        return chunks

    def get_chunk_summary(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Возвращает статистику по чанкам"""
        if not chunks:
            return {}

        total_chars = sum(len(chunk.text) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0

        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'smallest_chunk': min(len(chunk.text) for chunk in chunks),
            'largest_chunk': max(len(chunk.text) for chunk in chunks),
            'sources': list(set(chunk.source_file for chunk in chunks))
        }


class SemanticChunker:
    """Альтернативный чанкер на основе семантической близости"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.base_chunker = DocumentChunker()

    def create_semantic_chunks(self, text: str, source_file: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Создает чанки на основе семантической близости (упрощенная версия)"""
        # Пока используем базовый чанкер, но можно расширить с использованием embeddings
        # для определения семантически связанных фрагментов

        # Сначала разбиваем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 1:
            return self.base_chunker.create_chunks(text, source_file, metadata)

        # Группируем предложения в семантически связанные блоки
        # Это упрощенная версия - в продакшене стоит использовать embeddings
        chunks_text = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Если добавление предложения превысит лимит, создаем новый чанк
            if current_length + len(sentence) > self.base_chunker.chunk_size and current_chunk:
                chunks_text.append(' '.join(current_chunk))
                # Оставляем последнее предложение для связности (overlap)
                current_chunk = current_chunk[-1:] if current_chunk else []
                current_length = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(sentence)
            current_length += len(sentence)

        # Добавляем последний чанк
        if current_chunk:
            chunks_text.append(' '.join(current_chunk))

        # Преобразуем в TextChunk объекты
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = hashlib.md5(f"{source_file}_{i}_{chunk_text[:100]}".encode()).hexdigest()

            # ИСПРАВЛЕНО: Правильно копируем ВСЕ метаданные и в семантический чанкер
            chunk_metadata = {}
            if metadata:
                chunk_metadata.update(metadata)

            chunk_specific_metadata = {
                'chunk_size': len(chunk_text),
                'sentences_count': len(re.split(r'[.!?]', chunk_text)),
                'chunking_method': 'semantic',
                'chunk_index': i,
                'chunk_id': chunk_id
            }

            chunk_metadata.update(chunk_specific_metadata)

            chunk = TextChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_file=source_file,
                chunk_index=i,
                metadata=chunk_metadata
            )

            chunks.append(chunk)

        return chunks