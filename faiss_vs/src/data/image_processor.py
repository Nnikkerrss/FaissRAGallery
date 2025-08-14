"""
Модуль для обработки изображений в RAG системе
"""

import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import io
import numpy as np
import logging

# Для OCR можно использовать:
# - EasyOCR (простой в установке)
# - Tesseract (более мощный, но требует установки)
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ✅ НОВОЕ: Импорты для CLIP
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


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


# ✅ НОВЫЙ КЛАСС: Мультимодальный процессор с CLIP
class MultiModalProcessor:
    """Процессор для мультимодальной обработки изображений и текста с CLIP"""

    def __init__(self, device: Optional[str] = None):
        """
        Инициализация мультимодального процессора

        Args:
            device: Устройство для вычислений ('cuda', 'cpu' или None для автоопределения)
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP не установлен. Установите: pip install git+https://github.com/openai/CLIP.git")

        # Определяем устройство
        if device is None:
            self.device = settings.get_device_for_processing()
        else:
            self.device = device

        logger.info(f"Инициализация MultiModalProcessor на устройстве: {self.device}")

        # Инициализируем CLIP модель
        self.clip_model = None
        self.clip_preprocess = None
        self.visual_embedding_dim = settings.VISUAL_EMBEDDING_DIMENSION

        # Текстовый процессор (существующий)
        self.text_processor = ImageProcessor()

        # Загружаем модель при первом использовании (lazy loading)
        self._initialize_clip_model()

    def _initialize_clip_model(self):
        """Инициализирует CLIP модель"""
        try:
            logger.info(f"Загружаем CLIP модель {settings.CLIP_MODEL}...")
            self.clip_model, self.clip_preprocess = clip.load(settings.CLIP_MODEL, device=self.device)
            self.clip_model.eval()  # Режим инференса
            logger.info(f"✅ CLIP модель загружена на {self.device}")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки CLIP модели: {e}")
            raise Exception(f"Не удалось загрузить CLIP модель: {e}")

    def create_visual_embedding(self, image_path: Path) -> np.ndarray:
        """
        Создает визуальный эмбеддинг изображения с помощью CLIP

        Args:
            image_path: Путь к изображению

        Returns:
            np.ndarray: Нормализованный визуальный вектор
        """
        try:
            # Загружаем и предобрабатываем изображение
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # Создаем эмбеддинг
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                # Нормализуем для cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Конвертируем в numpy
            visual_vector = image_features.cpu().numpy().flatten()

            logger.debug(f"Создан визуальный вектор для {image_path.name}: {visual_vector.shape}")
            return visual_vector.astype(np.float32)

        except Exception as e:
            logger.error(f"Ошибка создания визуального эмбеддинга для {image_path}: {e}")
            # Возвращаем нулевой вектор в случае ошибки
            return np.zeros(self.visual_embedding_dim, dtype=np.float32)

    def create_text_embedding_for_image(self, image_path: Path, metadata: Dict[str, Any] = None) -> str:
        """
        Создает текстовое описание изображения (существующая логика)

        Args:
            image_path: Путь к изображению
            metadata: Метаданные из JSON

        Returns:
            str: Текстовое описание для векторизации
        """
        return self.text_processor.create_image_embedding_text(image_path, metadata)

    def process_image_multimodal(self, image_path: Path, metadata: Dict[str, Any] = None) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """
        Полная мультимодальная обработка изображения

        Args:
            image_path: Путь к изображению
            metadata: Исходные метаданные

        Returns:
            Tuple[str, np.ndarray, Dict]: (текстовое_описание, визуальный_вектор, обновленные_метаданные)
        """
        # 1. Создаем текстовое описание
        text_description = self.create_text_embedding_for_image(image_path, metadata)

        # 2. Создаем визуальный эмбеддинг
        visual_vector = self.create_visual_embedding(image_path)

        # 3. Обновляем метаданные
        updated_metadata = metadata.copy() if metadata else {}

        # Получаем размер изображения
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                updated_metadata.update({
                    'image_width': width,
                    'image_height': height,
                    'image_format': img.format,
                    'image_mode': img.mode
                })
        except Exception as e:
            logger.warning(f"Не удалось получить размеры изображения {image_path}: {e}")

        # Добавляем информацию о векторизации
        updated_metadata.update({
            'has_visual_embedding': True,
            'visual_embedding_dim': self.visual_embedding_dim,
            'visual_model': f'CLIP-{settings.CLIP_MODEL}',
            'processing_device': self.device,
            'multimodal_processing': True
        })

        logger.info(f"Мультимодальная обработка {image_path.name}: текст={len(text_description)} символов, вектор={visual_vector.shape}")

        return text_description, visual_vector, updated_metadata

    def search_by_text_description(self, text_query: str) -> np.ndarray:
        """
        Создает визуальный запрос из текстового описания

        Args:
            text_query: Текстовое описание ("красивый фасад здания")

        Returns:
            np.ndarray: Визуальный вектор, соответствующий описанию
        """
        try:
            # Токенизируем текст
            text_input = clip.tokenize([text_query]).to(self.device)

            # Создаем текстовый эмбеддинг в визуальном пространстве
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            visual_query = text_features.cpu().numpy().flatten()

            logger.debug(f"Создан визуальный запрос из текста '{text_query}': {visual_query.shape}")
            return visual_query.astype(np.float32)

        except Exception as e:
            logger.error(f"Ошибка создания визуального запроса из текста '{text_query}': {e}")
            return np.zeros(self.visual_embedding_dim, dtype=np.float32)

    def get_image_categories(self, image_path: Path, candidate_categories: List[str] = None) -> Dict[str, float]:
        """
        Определяет категории изображения с помощью CLIP

        Args:
            image_path: Путь к изображению
            candidate_categories: Список возможных категорий для классификации

        Returns:
            Dict[str, float]: Словарь {категория: уверенность}
        """
        if candidate_categories is None:
            candidate_categories = [
                "фасад здания", "внутренний интерьер", "техническая схема",
                "документ", "чертеж", "план", "фотография человека",
                "логотип", "текст документа", "электрическая схема"
            ]

        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # Подготавливаем текстовые описания
            text_descriptions = [f"Изображение {category}" for category in candidate_categories]
            text_input = clip.tokenize(text_descriptions).to(self.device)

            # Вычисляем сходство
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)

                # Нормализуем
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Вычисляем сходство (cosine similarity)
                similarity = (image_features @ text_features.T).squeeze(0)

                # Применяем softmax для получения вероятностей
                probabilities = torch.softmax(similarity, dim=0)

            # Создаем словарь результатов
            results = {}
            for category, prob in zip(candidate_categories, probabilities.cpu().numpy()):
                results[category] = float(prob)

            # Сортируем по убыванию вероятности
            results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

            logger.debug(f"Категории для {image_path.name}: {list(results.keys())[:3]}")
            return results

        except Exception as e:
            logger.error(f"Ошибка определения категорий для {image_path}: {e}")
            return {}

    def cleanup_gpu_memory(self):
        """Очищает GPU память после обработки"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU память очищена")

    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о загруженной модели"""
        return {
            'model_name': f'CLIP {settings.CLIP_MODEL}',
            'device': self.device,
            'visual_embedding_dim': self.visual_embedding_dim,
            'cuda_available': torch.cuda.is_available() if CLIP_AVAILABLE else False,
            'model_loaded': self.clip_model is not None,
            'clip_available': CLIP_AVAILABLE
        }


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


def check_dependencies():
    """Проверяет доступность зависимостей для мультимодального поиска"""
    status = {
        'ocr_available': OCR_AVAILABLE,
        'clip_available': CLIP_AVAILABLE,
        'torch_available': False,
        'cuda_available': False
    }

    if CLIP_AVAILABLE:
        try:
            import torch
            status['torch_available'] = True
            status['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            pass

    return status


# Инициализация при импорте
DEPENDENCIES = check_dependencies()

if not DEPENDENCIES['clip_available']:
    logger.warning("CLIP не установлен. Мультимодальный поиск недоступен.")
    logger.info("Установите: pip install git+https://github.com/openai/CLIP.git")

if DEPENDENCIES['cuda_available']:
    logger.info("✅ CUDA доступна для ускорения обработки изображений")
else:
    logger.info("ℹ️ CUDA недоступна, будет использоваться CPU")