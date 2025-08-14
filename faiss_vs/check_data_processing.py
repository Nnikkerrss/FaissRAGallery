#!/usr/bin/env python3
"""
Проверка качества обработки данных ПЕРЕД переиндексацией
Сохрани как: faiss_vs/check_data_processing.py
"""

import json
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.loaders import DocumentLoader
from src.data.chunkers import DocumentChunker


def check_json_data_quality(client_id: str):
    """
    📊 Проверяем качество исходных JSON данных
    """
    print("📊 ПРОВЕРКА КАЧЕСТВА ИСХОДНЫХ JSON ДАННЫХ")
    print("=" * 50)

    # Ищем JSON файл клиента
    from src.config import settings
    client_dir = settings.FAISS_INDEX_DIR / "clients" / client_id
    json_file = client_dir / "data.json"

    if not json_file.exists():
        print(f"❌ JSON файл не найден: {json_file}")
        return None

    print(f"📄 Загружаем: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'result' not in data:
        print("❌ Нет поля 'result' в JSON")
        return None

    documents = data['result']
    print(f"📈 Всего документов: {len(documents)}")

    # Анализируем качество метаданных
    stats = {
        'total': len(documents),
        'with_description': 0,
        'empty_description': 0,
        'with_parent': 0,
        'empty_parent': 0,
        'pdf_files': 0,
        'other_files': 0
    }

    problematic_docs = []
    good_examples = []

    print(f"\n📋 Анализ метаданных:")

    for i, doc in enumerate(documents):
        # Проверяем описание
        description = doc.get('Description', '')
        if description and description.strip():
            stats['with_description'] += 1
            if len(good_examples) < 3:
                good_examples.append({
                    'index': i,
                    'description': description,
                    'parent': doc.get('Parent', ''),
                    'url': doc.get('ID', '')
                })
        else:
            stats['empty_description'] += 1
            problematic_docs.append({
                'index': i,
                'problem': 'Пустое описание',
                'url': doc.get('ID', ''),
                'parent': doc.get('Parent', '')
            })

        # Проверяем категорию
        parent = doc.get('Parent', '')
        if parent and parent.strip():
            stats['with_parent'] += 1
        else:
            stats['empty_parent'] += 1

        # Проверяем тип файла
        url = doc.get('ID', '')
        if url.lower().endswith('.pdf'):
            stats['pdf_files'] += 1
        else:
            stats['other_files'] += 1

    # Выводим статистику
    print(
        f"   📝 С описанием: {stats['with_description']}/{stats['total']} ({stats['with_description'] / stats['total'] * 100:.1f}%)")
    print(
        f"   📁 С категорией: {stats['with_parent']}/{stats['total']} ({stats['with_parent'] / stats['total'] * 100:.1f}%)")
    print(f"   📄 PDF файлов: {stats['pdf_files']}")
    print(f"   📎 Других файлов: {stats['other_files']}")

    # Показываем хорошие примеры
    if good_examples:
        print(f"\n✅ ХОРОШИЕ ПРИМЕРЫ метаданных:")
        for example in good_examples:
            print(f"   {example['index']}. {example['description']}")
            print(f"      📁 {example['parent']}")
            print(f"      🔗 {example['url'][:60]}...")

    # Показываем проблемы
    if problematic_docs:
        print(f"\n⚠️ ПРОБЛЕМНЫЕ ДОКУМЕНТЫ:")
        for problem in problematic_docs[:5]:  # Первые 5
            print(f"   {problem['index']}. {problem['problem']}")
            print(f"      📁 {problem['parent']}")
            print(f"      🔗 {problem['url'][:60]}...")

    return documents


def test_loader_processing(documents_sample):
    """
    🔧 Тестируем обработку через DocumentLoader
    """
    print(f"\n🔧 ТЕСТИРОВАНИЕ ОБРАБОТКИ ЧЕРЕЗ DocumentLoader")
    print("=" * 55)

    # Создаем тестовый JSON файл
    test_data = {
        'result': documents_sample[:3]  # Берем первые 3 документа
    }

    test_file = Path("test_documents.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"📄 Создан тестовый файл: {test_file}")

    # Тестируем DocumentLoader
    try:
        loader = DocumentLoader()

        print(f"🔄 Обрабатываем через DocumentLoader...")

        # НЕ скачиваем файлы, только проверяем метаданные
        documents = loader.load_from_json(str(test_file))

        print(f"📊 Загружено документов: {len(documents)}")

        # Симулируем обработку метаданных
        for i, doc_info in enumerate(documents):
            print(f"\n📋 Документ {i}:")
            print(f"   🔗 URL: {doc_info.get('ID', 'НЕТ')}")

            # Проверяем как будут обработаны метаданные
            metadata = {}

            # Копируем все поля (как в реальном коде)
            for key, value in doc_info.items():
                if key not in ['url', 'filename', 'ID']:
                    normalized_key = key.lower()
                    metadata[normalized_key] = value

            # Добавляем маппинги
            if 'description' in metadata and metadata['description']:
                metadata['title'] = metadata['description']

            if 'parent' in metadata and metadata['parent']:
                metadata['category'] = metadata['parent']

            print(f"   📝 Исходное описание: {doc_info.get('Description', 'НЕТ')}")
            print(f"   📝 Финальный title: {metadata.get('title', 'НЕТ')}")
            print(f"   📁 Исходный Parent: {doc_info.get('Parent', 'НЕТ')}")
            print(f"   📁 Финальная category: {metadata.get('category', 'НЕТ')}")
            print(f"   🗂️ Всего полей метаданных: {len(metadata)}")

            # Проверяем проблемы
            if not metadata.get('title'):
                print(f"   ❌ ПРОБЛЕМА: Нет title!")
            if not metadata.get('category'):
                print(f"   ⚠️ ПРОБЛЕМА: Нет category!")

    except Exception as e:
        print(f"❌ Ошибка тестирования loader: {e}")
    finally:
        # Удаляем тестовый файл
        if test_file.exists():
            test_file.unlink()


def test_chunker_enhancement(sample_text: str, metadata: dict):
    """
    ✂️ Тестируем обогащение чанков
    """
    print(f"\n✂️ ТЕСТИРОВАНИЕ ОБОГАЩЕНИЯ ЧАНКОВ")
    print("=" * 40)

    chunker = DocumentChunker()

    print(f"📝 Исходный текст ({len(sample_text)} символов):")
    print(f"   {sample_text[:150]}...")

    print(f"\n🏷️ Метаданные:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")

    # Проверяем есть ли метод обогащения
    if hasattr(chunker, 'create_enhanced_chunk_text'):
        print(f"\n✅ Метод create_enhanced_chunk_text найден!")

        try:
            enhanced_text = chunker.create_enhanced_chunk_text(sample_text, metadata)

            print(f"🚀 Обогащенный текст ({len(enhanced_text)} символов):")
            print(f"   {enhanced_text[:300]}...")

            # Проверяем содержит ли обогащенный текст ключевые слова
            keywords_found = []
            if metadata.get('title') and metadata['title'] in enhanced_text:
                keywords_found.append('title')
            if metadata.get('category') and metadata['category'] in enhanced_text:
                keywords_found.append('category')

            print(f"✅ Ключевые слова в тексте: {keywords_found}")

            if len(keywords_found) > 0:
                print(f"✅ ОБОГАЩЕНИЕ РАБОТАЕТ!")
                return True
            else:
                print(f"❌ ОБОГАЩЕНИЕ НЕ РАБОТАЕТ!")
                return False

        except Exception as e:
            print(f"❌ Ошибка обогащения: {e}")
            return False
    else:
        print(f"❌ Метод create_enhanced_chunk_text НЕ НАЙДЕН!")
        print(f"💡 Нужно добавить обогащение в chunkers.py")
        return False


def recommend_fixes(documents):
    """
    💡 Рекомендации по исправлению
    """
    print(f"\n💡 РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ")
    print("=" * 40)

    # Считаем проблемы
    empty_descriptions = sum(1 for doc in documents
                             if not doc.get('Description', '').strip())
    empty_parents = sum(1 for doc in documents
                        if not doc.get('Parent', '').strip())

    print(f"📊 Найдено проблем:")
    print(f"   📝 Пустых описаний: {empty_descriptions}/{len(documents)}")
    print(f"   📁 Пустых категорий: {empty_parents}/{len(documents)}")

    if empty_descriptions > len(documents) * 0.3:
        print(f"\n❌ КРИТИЧНО: Слишком много пустых описаний!")
        print(f"   🛠️ Исправления:")
        print(f"   1. Проверь источник данных в 1С")
        print(f"   2. Добавь fallback в loaders.py:")
        print(f"      if not metadata.get('title'):")
        print(f"          metadata['title'] = filename  # Используй имя файла")

    if empty_parents > len(documents) * 0.2:
        print(f"\n⚠️ ПРОБЛЕМА: Много документов без категории!")
        print(f"   🛠️ Добавь в loaders.py:")
        print(f"      metadata['category'] = metadata.get('category') or 'Прочие документы'")

    print(f"\n🔧 ОБЯЗАТЕЛЬНЫЕ ИСПРАВЛЕНИЯ:")
    print(f"   1. ✅ Исправить loaders.py - правильный маппинг метаданных")
    print(f"   2. ✅ Добавить в chunkers.py - обогащение текста")
    print(f"   3. ✅ Исправить faiss_manager.py - умный поиск")
    print(f"   4. ✅ Только потом переиндексировать!")


def main():
    if len(sys.argv) < 2:
        print("Использование: python check_data_processing.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]

    try:
        # 1. Проверяем исходные JSON данные
        documents = check_json_data_quality(client_id)

        if not documents:
            print("❌ Не удалось загрузить данные")
            return

        # 2. Тестируем обработку через loader
        test_loader_processing(documents)

        # 3. Тестируем обогащение чанков
        sample_metadata = {
            'title': 'Спецификация окон и дверей',
            'description': 'Спецификация окон и дверей',
            'category': 'Файлы по архитектуре',
            'filename': 'spec_okna_dveri.pdf'
        }

        sample_text = "Данный документ содержит спецификацию оконных и дверных блоков для объекта строительства."

        enhancement_works = test_chunker_enhancement(sample_text, sample_metadata)

        # 4. Даем рекомендации
        recommend_fixes(documents)

        print(f"\n🎯 ФИНАЛЬНЫЕ ВЫВОДЫ:")
        print(f"   📊 Данные загружены: ✅")
        print(f"   🔧 Обогащение работает: {'✅' if enhancement_works else '❌'}")

        if enhancement_works:
            print(f"\n✅ МОЖНО ПЕРЕИНДЕКСИРОВАТЬ!")
            print(f"   python quick_cleanup.py {client_id}")
            print(f"   # Потом создать индекс заново")
        else:
            print(f"\n❌ СНАЧАЛА ИСПРАВЬ КОД, потом переиндексируй!")
            print(f"   1. Добавь обогащение в chunkers.py")
            print(f"   2. Исправь поиск в faiss_manager.py")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()