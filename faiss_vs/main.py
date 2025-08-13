"""
FAISS Image Indexer with LangChain
Проект для загрузки изображений из JSON и их индексации в FAISS
"""

import json
import os
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import hashlib
from datetime import datetime

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Image processing
from PIL import Image
import base64
from io import BytesIO


class ImageIndexer:
    """Класс для индексации изображений в FAISS"""

    def __init__(self,
                 data_dir: str = "data",
                 images_dir: str = "images",
                 faiss_index_dir: str = "faiss_index",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Инициализация индексера

        Args:
            data_dir: Директория для данных
            images_dir: Поддиректория для изображений
            faiss_index_dir: Директория для FAISS индекса
            embedding_model: Модель для создания эмбеддингов
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / images_dir
        self.faiss_index_dir = self.data_dir / faiss_index_dir

        # Создаем директории
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.faiss_index_dir.mkdir(exist_ok=True)

        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Инициализация эмбеддингов
        self.embeddings = self._init_embeddings(embedding_model)

        # Сессия для запросов
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _init_embeddings(self, model_name: str):
        """Инициализация модели эмбеддингов"""
        try:
            # Попробуем использовать OpenAI (если есть API ключ)
            if os.getenv('OPENAI_API_KEY'):
                self.logger.info("Используем OpenAI embeddings")
                return OpenAIEmbeddings()
            else:
                # Используем HuggingFace
                self.logger.info(f"Используем HuggingFace embeddings: {model_name}")
                return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            self.logger.warning(f"Ошибка инициализации эмбеддингов: {e}")
            # Fallback на простую модель
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_json_data(self, url: str) -> List[Dict[str, Any]]:
        """Загрузка данных из JSON по URL"""
        try:
            self.logger.info(f"Загружаем данные из {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Если JSON не является списком, оборачиваем в список
            if not isinstance(data, list):
                data = [data]

            self.logger.info(f"Загружено {len(data)} записей")
            return data

        except requests.RequestException as e:
            self.logger.error(f"Ошибка загрузки JSON: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка парсинга JSON: {e}")
            raise

    def download_image(self, image_url: str, filename: str) -> Optional[str]:
        """Скачивание изображения по URL"""
        try:
            response = self.session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()

            # Проверяем, что это изображение
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                self.logger.warning(f"URL не является изображением: {image_url}")
                return None

            # Сохраняем изображение
            image_path = self.images_dir / filename
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Проверяем, что изображение корректно
            try:
                with Image.open(image_path) as img:
                    img.verify()
                self.logger.info(f"Изображение сохранено: {filename}")
                return str(image_path)
            except Exception as e:
                self.logger.error(f"Поврежденное изображение {filename}: {e}")
                image_path.unlink(missing_ok=True)
                return None

        except Exception as e:
            self.logger.error(f"Ошибка скачивания изображения {image_url}: {e}")
            return None

    def generate_filename(self, url: str, item_data: Dict) -> str:
        """Генерация имени файла на основе URL и данных"""
        # Получаем расширение из URL
        parsed_url = urlparse(url)
        ext = Path(parsed_url.path).suffix

        if not ext:
            ext = '.jpg'  # По умолчанию

        # Создаем хеш на основе URL и некоторых данных
        hash_data = f"{url}_{item_data.get('id', '')}_{item_data.get('title', '')}"
        filename_hash = hashlib.md5(hash_data.encode()).hexdigest()[:12]

        return f"{filename_hash}{ext}"

    def extract_image_urls(self, item: Dict[str, Any]) -> List[str]:
        """Извлечение URL изображений из элемента данных"""
        image_urls = []

        # Возможные поля с изображениями
        image_fields = ['image', 'images', 'photo', 'photos', 'picture', 'pictures',
                        'thumbnail', 'avatar', 'cover', 'banner']

        for field in image_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str) and value.startswith(('http://', 'https://')):
                    image_urls.append(value)
                elif isinstance(value, list):
                    for img in value:
                        if isinstance(img, str) and img.startswith(('http://', 'https://')):
                            image_urls.append(img)
                        elif isinstance(img, dict) and 'url' in img:
                            if img['url'].startswith(('http://', 'https://')):
                                image_urls.append(img['url'])

        return list(set(image_urls))  # Убираем дубликаты

    def process_data(self, json_url: str) -> List[Document]:
        """Основная функция обработки данных"""
        # Загружаем JSON данные
        data = self.load_json_data(json_url)

        documents = []

        for idx, item in enumerate(data):
            try:
                # Извлекаем URL изображений
                image_urls = self.extract_image_urls(item)

                if not image_urls:
                    self.logger.warning(f"Изображения не найдены в элементе {idx}")
                    continue

                # Скачиваем изображения
                downloaded_images = []
                for img_url in image_urls:
                    filename = self.generate_filename(img_url, item)
                    local_path = self.download_image(img_url, filename)
                    if local_path:
                        downloaded_images.append({
                            'original_url': img_url,
                            'local_path': local_path,
                            'filename': filename
                        })

                if not downloaded_images:
                    self.logger.warning(f"Не удалось скачать изображения для элемента {idx}")
                    continue

                # Создаем текстовое описание для эмбеддинга
                text_content = self.create_text_description(item, downloaded_images)

                # Создаем Document для LangChain
                metadata = {
                    'item_index': idx,
                    'images': downloaded_images,
                    'original_data': item,
                    'timestamp': datetime.now().isoformat()
                }

                doc = Document(
                    page_content=text_content,
                    metadata=metadata
                )

                documents.append(doc)
                self.logger.info(f"Обработан элемент {idx}, изображений: {len(downloaded_images)}")

            except Exception as e:
                self.logger.error(f"Ошибка обработки элемента {idx}: {e}")
                continue

        return documents

    def create_text_description(self, item: Dict, images: List[Dict]) -> str:
        """Создание текстового описания для поиска"""
        description_parts = []

        # Основные текстовые поля
        text_fields = ['title', 'name', 'description', 'content', 'text', 'caption']

        for field in text_fields:
            if field in item and item[field]:
                description_parts.append(f"{field}: {item[field]}")

        # Информация об изображениях
        if images:
            image_info = f"Количество изображений: {len(images)}"
            filenames = [img['filename'] for img in images]
            image_info += f", Файлы: {', '.join(filenames)}"
            description_parts.append(image_info)

        # Дополнительные метаданные
        for key, value in item.items():
            if key not in text_fields + ['image', 'images', 'photo', 'photos'] and \
                    isinstance(value, (str, int, float)) and len(str(value)) < 100:
                description_parts.append(f"{key}: {value}")

        return "\n".join(description_parts)

    def create_faiss_index(self, documents: List[Document]) -> FAISS:
        """Создание FAISS индекса"""
        if not documents:
            raise ValueError("Нет документов для индексации")

        self.logger.info(f"Создаем FAISS индекс для {len(documents)} документов")

        # Создаем векторное хранилище
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # Сохраняем индекс
        vectorstore.save_local(str(self.faiss_index_dir))
        self.logger.info(f"FAISS индекс сохранен в {self.faiss_index_dir}")

        return vectorstore

    def load_faiss_index(self) -> Optional[FAISS]:
        """Загрузка существующего FAISS индекса"""
        try:
            vectorstore = FAISS.load_local(
                str(self.faiss_index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info("FAISS индекс загружен")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Ошибка загрузки FAISS индекса: {e}")
            return None

    def search_similar(self, vectorstore: FAISS, query: str, k: int = 5) -> List[Dict]:
        """Поиск похожих изображений по текстовому запросу"""
        results = vectorstore.similarity_search(query, k=k)

        search_results = []
        for doc in results:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'images': doc.metadata.get('images', []),
                'original_data': doc.metadata.get('original_data', {})
            }
            search_results.append(result)

        return search_results


def main():
    """Основная функция запуска"""
    # URL для тестирования (замените на ваш)
    JSON_URL = "https://jsonfile.com"  # Замените на реальный URL

    # Создаем индексер
    indexer = ImageIndexer()

    try:
        # Обрабатываем данные
        print("Обработка данных из JSON...")
        documents = indexer.process_data(JSON_URL)

        if not documents:
            print("Не найдено документов для индексации")
            return

        # Создаем FAISS индекс
        print("Создание FAISS индекса...")
        vectorstore = indexer.create_faiss_index(documents)

        # Пример поиска
        print("\nПример поиска:")
        query = "красивое изображение"  # Замените на ваш запрос
        results = indexer.search_similar(vectorstore, query, k=3)

        for i, result in enumerate(results, 1):
            print(f"\nРезультат {i}:")
            print(f"Описание: {result['content'][:200]}...")
            print(f"Изображений: {len(result['images'])}")
            for img in result['images']:
                print(f"  - {img['filename']} ({img['original_url']})")

        print(f"\nВсего обработано: {len(documents)} документов")
        print(f"Изображения сохранены в: {indexer.images_dir}")
        print(f"FAISS индекс сохранен в: {indexer.faiss_index_dir}")

    except Exception as e:
        print(f"Ошибка выполнения: {e}")
        logging.error(f"Ошибка в main: {e}", exc_info=True)


if __name__ == "__main__":
    main()