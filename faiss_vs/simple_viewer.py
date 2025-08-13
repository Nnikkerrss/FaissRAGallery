#!/usr/bin/env python3
"""
Простой скрипт для просмотра сохраненного содержимого
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor


def main():
    """Показывает что сохранено в индексе"""
    print("🔍 Что сохранено в вашем RAG индексе")
    print("=" * 50)

    # Создаем процессор
    processor = DocumentProcessor()

    # Загружаем сохраненный индекс
    loaded = processor.faiss_manager.load_index()

    if not loaded:
        print("❌ Сохраненный индекс не найден!")
        print("Сначала запустите: python test_your_documents.py")
        return

    # Получаем статистику
    stats = processor.get_index_statistics()

    print(f"📊 Общая статистика:")
    print(f"   📄 Документов обработано: {stats.get('sources_count', 0)}")
    print(f"   🧩 Всего фрагментов (чанков): {stats.get('total_chunks', 0)}")
    print(f"   🔢 Векторов в индексе: {stats.get('total_vectors', 0)}")
    print(f"   📝 Всего символов: {stats.get('total_characters', 0):,}")

    # Показываем какие файлы обработаны
    if 'sources_distribution' in stats:
        print(f"\n📁 Обработанные файлы:")
        for i, (filename, chunks_count) in enumerate(stats['sources_distribution'].items(), 1):
            print(f"   {i}. {filename}")
            print(f"      Разбито на {chunks_count} фрагментов")

    # Показываем категории
    if 'categories_distribution' in stats:
        print(f"\n📂 По категориям:")
        for category, count in stats['categories_distribution'].items():
            print(f"   {category}: {count} фрагментов")

    # Тестируем поиск
    print(f"\n🔍 Тест поиска:")
    test_queries = ["электрика", "альбом", "проект"]

    for query in test_queries:
        results = processor.search_documents(query, k=2)
        print(f"\n   Поиск '{query}':")
        if results:
            for j, result in enumerate(results, 1):
                print(f"     {j}. {result['source_file']} (точность: {result['score']:.2f})")
                # Показываем короткий фрагмент найденного текста
                preview = result['text'].replace('\n', ' ')[:100] + "..."
                print(f"        «{preview}»")
        else:
            print(f"     Ничего не найдено")

    # Показываем пример чанка
    print(f"\n📖 Пример сохраненного фрагмента:")
    all_chunks = processor.faiss_manager.get_all_chunks()
    if all_chunks:
        example_chunk = all_chunks[0]
        print(f"   Файл: {example_chunk['source_file']}")
        print(f"   Размер: {len(example_chunk['text'])} символов")
        print(f"   Начало текста:")
        preview_text = example_chunk['text'][:300].replace('\n', '\n   ')
        print(f"   «{preview_text}...»")

    print(f"\n✅ Индекс работает! Можно использовать для поиска.")
    print(f"💡 Для поиска используйте: processor.search_documents('ваш запрос')")


if __name__ == "__main__":
    main()