#!/usr/bin/env python3
"""
Скрипт для диагностики проблем с поиском через API
Сохрани как: faiss_vs/debug_search_api.py
"""

import sys
import requests
import json
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor


def debug_search_api(client_id: str):
    """
    🔍 Диагностируем проблемы с поиском через API
    """
    print("🔍 ДИАГНОСТИКА ПОИСКА ЧЕРЕЗ API")
    print("=" * 60)

    # 1. Проверяем индекс напрямую
    print("📊 Шаг 1: Проверяем индекс напрямую")
    try:
        processor = DocumentProcessor(client_id=client_id)
        stats = processor.get_index_statistics()

        print(f"   Статус индекса: {stats.get('status', 'unknown')}")
        print(f"   Всего векторов: {stats.get('total_vectors', 0)}")
        print(f"   Всего чанков: {stats.get('total_chunks', 0)}")
        print(f"   Источников: {stats.get('sources_count', 0)}")

        if stats.get('total_chunks', 0) == 0:
            print("   ❌ ПРОБЛЕМА: Индекс пустой!")
            return False
        else:
            print("   ✅ Индекс содержит данные")

    except Exception as e:
        print(f"   ❌ Ошибка доступа к индексу: {e}")
        return False

    # 2. Тестируем прямой поиск
    print("\n🔍 Шаг 2: Тестируем прямой поиск")
    test_queries = ["фасад", "фото", "архитектура", "здание"]

    for query in test_queries:
        try:
            results = processor.search_documents(query, k=3)
            print(f"   Запрос '{query}': найдено {len(results)} результатов")

            if results:
                best = results[0]
                print(f"      Лучший: {best['source_file']} (score: {best.get('score', 0):.3f})")
                metadata = best.get('metadata', {})
                print(f"      Описание: {metadata.get('title', 'НЕТ')}")

        except Exception as e:
            print(f"   ❌ Ошибка поиска '{query}': {e}")

    # 3. Проверяем API endpoint
    print("\n🌐 Шаг 3: Проверяем API endpoint")
    api_url = "http://127.0.0.1:8000/faiss/search"

    test_payload = {
        "client_id": client_id,
        "query": "фасад"
    }

    try:
        response = requests.post(api_url, json=test_payload, timeout=10)
        print(f"   HTTP статус: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Результатов через API: {data.get('results_count', 0)}")

            if data.get('results_count', 0) == 0:
                print("   ⚠️ API возвращает пустые результаты")
                print("   💡 Проблема в routes.py или логике API")
            else:
                print("   ✅ API работает корректно")

        else:
            print(f"   ❌ API ошибка: {response.status_code}")
            print(f"   Ответ: {response.text}")

    except Exception as e:
        print(f"   ❌ Ошибка подключения к API: {e}")

    # 4. Показываем примеры документов
    print("\n📚 Шаг 4: Примеры документов в индексе")
    try:
        all_chunks = processor.faiss_manager.get_all_chunks()

        if all_chunks:
            print(f"   Всего чанков: {len(all_chunks)}")

            # Показываем файлы
            files = set()
            categories = set()

            for chunk in all_chunks:
                if chunk.get('source_file'):
                    files.add(chunk['source_file'])
                metadata = chunk.get('metadata', {})
                if metadata.get('category'):
                    categories.add(metadata['category'])

            print(f"   📁 Уникальных файлов: {len(files)}")
            print(f"   📂 Категорий: {len(categories)}")

            print("\n   📄 Примеры файлов:")
            for i, filename in enumerate(list(files)[:5]):
                print(f"      {i + 1}. {filename}")

            print("\n   📂 Категории:")
            for category in list(categories)[:5]:
                print(f"      - {category}")

            # Ищем файлы связанные с архитектурой/фасадами
            print("\n   🏗️ Файлы связанные с архитектурой:")
            arch_files = [f for f in files if any(word in f.lower()
                                                  for word in ['архитект', 'фасад', 'план', 'проект', 'концепц'])]

            if arch_files:
                for filename in arch_files[:3]:
                    print(f"      📐 {filename}")
            else:
                print("      ❌ Не найдено файлов по архитектуре")

        else:
            print("   ❌ Нет чанков в индексе")

    except Exception as e:
        print(f"   ❌ Ошибка получения чанков: {e}")

    return True


def check_routes_py():
    """
    📝 Проверяем файл routes.py
    """
    print("\n📝 Шаг 5: Проверяем routes.py")

    routes_path = Path("routes.py")
    if not routes_path.exists():
        routes_path = Path("faiss_vs/routes.py")

    if routes_path.exists():
        print(f"   📄 Найден файл: {routes_path}")

        with open(routes_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Проверяем наличие search endpoint
        if '/faiss/search' in content:
            print("   ✅ Endpoint /faiss/search найден")

            # Ищем реализацию
            if 'search_documents' in content:
                print("   ✅ Вызов search_documents найден")
            else:
                print("   ⚠️ Возможно нет вызова search_documents")

            if 'DocumentProcessor' in content:
                print("   ✅ Import DocumentProcessor найден")
            else:
                print("   ❌ НЕТ import DocumentProcessor!")

        else:
            print("   ❌ Endpoint /faiss/search НЕ НАЙДЕН!")

    else:
        print("   ❌ Файл routes.py не найден!")


def suggest_fixes():
    """
    💡 Предлагаем исправления
    """
    print("\n💡 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
    print("=" * 40)

    print("1. 🔄 Перезапустите сервер:")
    print("   cd /path/to/project")
    print("   python main.py")

    print("\n2. 📝 Проверьте routes.py:")
    print("   - Есть ли endpoint /faiss/search")
    print("   - Импортирован ли DocumentProcessor")
    print("   - Вызывается ли search_documents")

    print("\n3. 🗂️ Проверьте индекс:")
    print("   python check_metadata.py 6a2502fa-caaa-11e3-9af3-e41f13beb1d2")

    print("\n4. 🔍 Тестируйте поиск напрямую:")
    print("   python -c \"")
    print("   from src.document_processor import DocumentProcessor")
    print("   p = DocumentProcessor('6a2502fa-caaa-11e3-9af3-e41f13beb1d2')")
    print("   print(p.search_documents('фасад', k=3))")
    print("   \"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python debug_search_api.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]

    try:
        success = debug_search_api(client_id)
        check_routes_py()

        if not success:
            suggest_fixes()

    except Exception as e:
        print(f"❌ Ошибка диагностики: {e}")
        import traceback

        traceback.print_exc()