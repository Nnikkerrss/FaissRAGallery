#!/usr/bin/env python3
"""
Скрипт для проверки метаданных в FAISS индексе
"""

import sys
import json
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor


def check_metadata(client_id: str):
    """Проверяет метаданные в индексе клиента"""
    print(f"🔍 Проверяем метаданные для клиента: {client_id}")
    print("=" * 60)

    processor = DocumentProcessor(client_id=client_id)

    # Получаем статистику индекса
    stats = processor.get_index_statistics()

    if stats.get('status') != 'ready':
        print(f"❌ Индекс не готов: {stats}")
        return

    print(f"📊 Статистика индекса:")
    print(f"   Векторов: {stats.get('total_vectors', 0)}")
    print(f"   Чанков: {stats.get('total_chunks', 0)}")
    print(f"   Источников: {stats.get('sources_count', 0)}")

    # Получаем все чанки
    all_chunks = processor.faiss_manager.get_all_chunks()

    if not all_chunks:
        print("❌ Нет чанков в индексе")
        return

    print(f"\n📋 Анализ метаданных ({len(all_chunks)} чанков):")

    # Анализируем метаданные
    metadata_fields = set()
    missing_fields = []
    field_stats = {}

    for i, chunk in enumerate(all_chunks):
        chunk_metadata = chunk.get('metadata', {})

        # Собираем все поля метаданных
        for field in chunk_metadata.keys():
            metadata_fields.add(field)

        # Статистика по полям
        for field, value in chunk_metadata.items():
            if field not in field_stats:
                field_stats[field] = {'count': 0, 'examples': []}
            field_stats[field]['count'] += 1
            if len(field_stats[field]['examples']) < 3:
                field_stats[field]['examples'].append(str(value)[:100])

        # Показываем первые несколько чанков подробно
        if i < 3:
            print(f"\n📄 Чанк {i}:")
            print(f"   Источник: {chunk.get('source_file', 'unknown')}")
            print(f"   Текст: {chunk.get('text', '')[:100]}...")
            print(f"   Метаданных: {len(chunk_metadata)}")

            print("   📝 Метаданные:")
            for key, value in chunk_metadata.items():
                value_str = str(value)[:50]
                if len(str(value)) > 50:
                    value_str += "..."
                print(f"      {key}: {value_str}")

    print(f"\n📊 Общая статистика метаданных:")
    print(f"   Уникальных полей: {len(metadata_fields)}")
    print(f"   Поля: {sorted(metadata_fields)}")

    print(f"\n📈 Статистика по полям:")
    for field, stats in sorted(field_stats.items()):
        coverage = (stats['count'] / len(all_chunks)) * 100
        print(f"   {field}:")
        print(f"      Покрытие: {stats['count']}/{len(all_chunks)} ({coverage:.1f}%)")
        print(f"      Примеры: {stats['examples']}")

    # Проверяем важные поля
    important_fields = ['title', 'description', 'category', 'parent', 'date', 'source_url', 'file_type']

    print(f"\n⚠️  Проверка важных полей:")
    for field in important_fields:
        if field in metadata_fields:
            coverage = (field_stats[field]['count'] / len(all_chunks)) * 100
            status = "✅" if coverage > 80 else "⚠️" if coverage > 50 else "❌"
            print(f"   {status} {field}: {coverage:.1f}% покрытие")
        else:
            print(f"   ❌ {field}: отсутствует")

    # Ищем проблемы
    print(f"\n🚨 Потенциальные проблемы:")

    # Чанки без базовых метаданных
    chunks_without_source = [c for c in all_chunks if not c.get('metadata', {}).get('source_url')]
    if chunks_without_source:
        print(f"   - {len(chunks_without_source)} чанков без source_url")

    chunks_without_category = [c for c in all_chunks if not c.get('metadata', {}).get('category')]
    if chunks_without_category:
        print(f"   - {len(chunks_without_category)} чанков без category")

    chunks_without_title = [c for c in all_chunks if not c.get('metadata', {}).get('title')]
    if chunks_without_title:
        print(f"   - {len(chunks_without_title)} чанков без title")

    return {
        'total_chunks': len(all_chunks),
        'metadata_fields': sorted(metadata_fields),
        'field_stats': field_stats
    }


def export_metadata_sample(client_id: str, output_file: str = None):
    """Экспортирует образец метаданных для анализа"""
    processor = DocumentProcessor(client_id=client_id)
    all_chunks = processor.faiss_manager.get_all_chunks()

    if not all_chunks:
        print("❌ Нет чанков для экспорта")
        return

    # Берем первые 5 чанков как образец
    sample_chunks = all_chunks[:5]

    sample_data = {
        'client_id': client_id,
        'total_chunks': len(all_chunks),
        'sample_count': len(sample_chunks),
        'chunks': []
    }

    for chunk in sample_chunks:
        sample_data['chunks'].append({
            'source_file': chunk.get('source_file'),
            'text_preview': chunk.get('text', '')[:200],
            'metadata': chunk.get('metadata', {})
        })

    if not output_file:
        output_file = f"metadata_sample_{client_id}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Образец метаданных сохранен в {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python check_metadata.py <client_id>")
        print("  python check_metadata.py <client_id> export")
        sys.exit(1)

    client_id = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "check"

    try:
        if command == "export":
            export_metadata_sample(client_id)
        else:
            result = check_metadata(client_id)

            if result:
                print(f"\n✅ Проверка завершена:")
                print(f"   Всего чанков: {result['total_chunks']}")
                print(f"   Полей метаданных: {len(result['metadata_fields'])}")

                # Предлагаем экспорт
                export_sample = input(f"\nЭкспортировать образец метаданных? (y/N): ")
                if export_sample.lower() == 'y':
                    export_metadata_sample(client_id)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()