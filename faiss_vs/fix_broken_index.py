#!/usr/bin/env python3
"""
Отладка ошибки NoneType в поиске
Сохрани как: faiss_vs/debug_none_error.py
"""

import sys
from pathlib import Path
import traceback

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor


def debug_none_error(client_id: str):
    """
    🐛 Отлаживаем ошибку NoneType
    """
    print("🐛 ОТЛАДКА ОШИБКИ NoneType")
    print("=" * 40)

    try:
        # 1. Создаем processor
        print("📝 Создаем DocumentProcessor...")
        processor = DocumentProcessor(client_id=client_id)
        print("✅ DocumentProcessor создан")

        # 2. Проверяем FAISS manager
        print("\n🔧 Проверяем FAISSManager...")
        manager = processor.faiss_manager
        print(f"   Client ID: {manager.client_id}")
        print(f"   Enable visual search: {manager.enable_visual_search}")

        # 3. Проверяем индексы
        print("\n📊 Проверяем индексы:")
        if hasattr(manager, 'text_index'):
            print(f"   text_index: {manager.text_index}")
            if manager.text_index:
                print(f"   text_index.ntotal: {manager.text_index.ntotal}")

        if hasattr(manager, 'index'):
            print(f"   index: {manager.index}")
            if manager.index:
                print(f"   index.ntotal: {manager.index.ntotal}")

        # 4. Проверяем метаданные
        print(f"\n📋 Метаданные:")
        print(f"   metadata: {len(manager.metadata)} элементов")
        print(f"   id_to_chunk_id: {len(manager.id_to_chunk_id)} элементов")

        if hasattr(manager, 'text_id_to_chunk_id'):
            print(f"   text_id_to_chunk_id: {len(manager.text_id_to_chunk_id)} элементов")

        # 5. Тестируем создание embeddings
        print(f"\n🧠 Тестируем создание embeddings...")
        try:
            test_embeddings = manager.create_embeddings(["тест"])
            print(f"   ✅ Embeddings созданы: shape={test_embeddings.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка создания embeddings: {e}")
            traceback.print_exc()
            return

        # 6. Тестируем поиск по шагам
        print(f"\n🔍 Тестируем поиск по шагам...")
        query = "электрика"

        try:
            print(f"   Шаг 1: Создаем embedding для '{query}'...")
            query_embedding = manager.create_embeddings([query])
            print(f"   ✅ Query embedding: shape={query_embedding.shape}")

            print(f"   Шаг 2: Определяем активный индекс...")
            if manager.enable_visual_search:
                active_index = manager.text_index
                print(f"   Активный: text_index ({active_index.ntotal if active_index else 'None'} векторов)")
            else:
                active_index = manager.index
                print(f"   Активный: index ({active_index.ntotal if active_index else 'None'} векторов)")

            if active_index is None:
                print(f"   ❌ ПРОБЛЕМА: активный индекс None!")
                return

            print(f"   Шаг 3: Выполняем FAISS поиск...")
            search_k = min(5, active_index.ntotal)
            scores, indices = active_index.search(query_embedding, search_k)

            print(f"   ✅ FAISS поиск выполнен:")
            print(f"      scores: {scores}")
            print(f"      indices: {indices}")
            print(f"      scores[0]: {scores[0] if scores is not None else 'None'}")
            print(f"      indices[0]: {indices[0] if indices is not None else 'None'}")

            if scores is None or indices is None:
                print(f"   ❌ ПРОБЛЕМА: scores или indices равны None!")
                return

            print(f"   Шаг 4: Обрабатываем результаты...")
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                print(f"      Результат {i}: score={score}, idx={idx}")

                if idx == -1:
                    print(f"         Пропускаем idx=-1")
                    continue

                # Определяем правильный маппинг
                if manager.enable_visual_search:
                    chunk_id = manager.text_id_to_chunk_id.get(idx)
                    print(f"         text_id_to_chunk_id[{idx}] = {chunk_id}")
                else:
                    chunk_id = manager.id_to_chunk_id.get(idx)
                    print(f"         id_to_chunk_id[{idx}] = {chunk_id}")

                if chunk_id and chunk_id in manager.metadata:
                    print(f"         ✅ Найден чанк: {chunk_id}")
                else:
                    print(f"         ❌ Чанк не найден в метаданных")

        except Exception as e:
            print(f"   ❌ Ошибка в поиске по шагам: {e}")
            traceback.print_exc()
            return

        # 7. Тестируем полный поиск через manager
        print(f"\n🎯 Тестируем полный поиск через manager...")
        try:
            results = manager.search(query, k=3)
            print(f"   ✅ Поиск через manager: {len(results)} результатов")

            for i, result in enumerate(results):
                print(f"      {i}: {result.get('source_file', 'unknown')} (score: {result.get('score', 0):.3f})")

        except Exception as e:
            print(f"   ❌ Ошибка в manager.search: {e}")
            traceback.print_exc()

        # 8. Тестируем поиск через processor
        print(f"\n🚀 Тестируем поиск через processor...")
        try:
            results = processor.search_documents(query, k=3)
            print(f"   ✅ Поиск через processor: {len(results)} результатов")

            for i, result in enumerate(results):
                print(f"      {i}: {result.get('source_file', 'unknown')} (score: {result.get('score', 0):.3f})")

        except Exception as e:
            print(f"   ❌ Ошибка в processor.search_documents: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()


def check_routes_py():
    """
    📝 Проверяем routes.py на предмет ошибок
    """
    print(f"\n📝 ПРОВЕРКА ROUTES.PY")
    print("=" * 30)

    routes_file = Path("faiss_vs/routes.py")

    if not routes_file.exists():
        print("❌ routes.py не найден!")
        return

    with open(routes_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ищем функцию search
    lines = content.split('\n')
    in_search_function = False
    search_function_lines = []

    for line in lines:
        if 'def search():' in line:
            in_search_function = True
            search_function_lines.append(line)
        elif in_search_function:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # Функция закончилась
                break
            search_function_lines.append(line)

    print("📄 Функция search() в routes.py:")
    for line in search_function_lines:
        print(f"   {line}")

    # Проверяем на подозрительные места
    search_code = '\n'.join(search_function_lines)

    if 'processor.search_documents' in search_code:
        print("✅ Вызов processor.search_documents найден")
    else:
        print("❌ НЕТ вызова processor.search_documents!")

    if 'try:' in search_code and 'except:' in search_code:
        print("✅ Есть обработка ошибок")
    else:
        print("⚠️ Нет обработки ошибок - это может быть причиной")


def main():
    if len(sys.argv) < 2:
        print("Использование: python debug_none_error.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]

    debug_none_error(client_id)
    check_routes_py()


if __name__ == "__main__":
    main()