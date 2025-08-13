"""
Модуль для обработки изображений в RAG системе
"""

import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import io

# Для OCR можно использовать:
# - EasyOCR (простой в установке)
# - Tesseract (более мощный, но требует установки)
try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class ImageProcessor:
    """Обработчик изображений для извлечения текста и описаний"""

    def __init__(self):
        self.ocr_reader = None
        if OCR_AVAILABLE:
            # Инициализируем OCR для русского и английского языков
            self.ocr_reader = easyocr.Reader(['ru', 'en'])

    def extract_text_from_image(self, image_path: Path) -> str:
        """Извлекает текст из изображения с помощью OCR"""
        if not OCR_AVAILABLE or not self.ocr_reader:
            return ""

        try:
            # Читаем изображение и извлекаем текст
            results = self.ocr_reader.readtext(str(image_path))

            # Объединяем весь найденный текст
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Берем только текст с высокой уверенностью
                    extracted_text.append(text)

            return " ".join(extracted_text)

        except Exception as e:
            print(f"Ошибка OCR для {image_path}: {e}")
            return ""

    def get_image_description(self, image_path: Path, metadata: Dict[str, Any] = None) -> str:
        """Создает описание изображения на основе метаданных и имени файла"""
        description_parts = []

        # Используем метаданные из JSON
        if metadata:
            if metadata.get('title'):
                description_parts.append(f"Название: {metadata['title']}")
            if metadata.get('description'):
                description_parts.append(f"Описание: {metadata['description']}")
            if metadata.get('category'):
                description_parts.append(f"Категория: {metadata['category']}")

        # Анализируем имя файла
        filename = image_path.stem
        description_parts.append(f"Файл: {filename}")

        # Получаем базовую информацию об изображении
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
                description_parts.append(f"Изображение {format_name} {width}x{height}")
        except Exception:
            description_parts.append("Изображение")

        # Извлекаем текст с помощью OCR
        ocr_text = self.extract_text_from_image(image_path)
        if ocr_text.strip():
            description_parts.append(f"Текст на изображении: {ocr_text}")

        return ". ".join(description_parts)

    def process_image_document(self, image_path: Path, metadata: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
        """
        Обрабатывает изображение как документ

        Returns:
            tuple: (текст_описание, обновленные_метаданные)
        """
        # Создаем текстовое описание изображения
        description = self.get_image_description(image_path, metadata)

        # Обновляем метаданные
        updated_metadata = metadata.copy() if metadata else {}
        updated_metadata.update({
            'document_type': 'image',
            'has_ocr': OCR_AVAILABLE,
            'ocr_text_length': len(self.extract_text_from_image(image_path)) if OCR_AVAILABLE else 0
        })

        return description, updated_metadata

    @staticmethod
    def is_image_file(file_path: Path) -> bool:
        """Проверяет, является ли файл изображением"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return file_path.suffix.lower() in image_extensions

    def create_image_embedding_text(self, image_path: Path, metadata: Dict[str, Any] = None) -> str:
        """
        Создает текст для создания embedding'а изображения
        Этот текст будет использован для векторизации
        """
        # Базовое описание
        text_parts = []

        if metadata:
            # Используем структурированные метаданные
            if metadata.get('description'):
                text_parts.append(metadata['description'])
            if metadata.get('category'):
                text_parts.append(f"Категория: {metadata['category']}")
            if metadata.get('parent'):
                text_parts.append(f"Раздел: {metadata['parent']}")

        # Добавляем имя файла (может содержать полезную информацию)
        filename_info = image_path.stem.replace('_', ' ').replace('-', ' ')
        text_parts.append(filename_info)

        # OCR текст (если доступен)
        if OCR_AVAILABLE:
            ocr_text = self.extract_text_from_image(image_path)
            if ocr_text.strip():
                text_parts.append(f"Текст на изображении: {ocr_text}")

        # Объединяем все части
        combined_text = ". ".join(filter(None, text_parts))

        # Если текста мало, добавляем общее описание
        if len(combined_text) < 50:
            combined_text += f". Изображение из документации проекта."

        return combined_text


# Пример конфигурации для установки EasyOCR
def install_ocr_instructions():
    """Выводит инструкции по установке OCR"""
    print("""
    Для работы с изображениями установите EasyOCR:

    pip install easyocr

    Альтернативно можно использовать Tesseract:

    # Ubuntu/Debian:
    sudo apt-get install tesseract-ocr tesseract-ocr-rus
    pip install pytesseract

    # macOS:
    brew install tesseract tesseract-lang
    pip install pytesseract

    # Windows:
    # Скачайте Tesseract с https://github.com/UB-Mannheim/tesseract/wiki
    pip install pytesseract
    """)