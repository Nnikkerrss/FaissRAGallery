#!/usr/bin/env python3
"""
Быстрая очистка данных RAG системы по client_id
"""

import sys
import shutil
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.config import settings


def quick_cleanup(client_id: str):
    """Быстро очищает данные по client_id"""
    print(f"🧹 Быстрая очистка RAG данных клиента: {client_id}")
    print("=" * 40)

    processor = DocumentProcessor(client_id=client_id)

    # ИСПРАВЛЕНО: Правильные пути к данным клиента
    client_docs_path = settings.DOCUMENTS_DIR / client_id
    client_faiss_path = settings.FAISS_INDEX_DIR / "clients" / client_id  # Добавлено "clients"

    if not client_docs_path.exists() and not client_faiss_path.exists():
        print(f"❌ Нет данных для клиента {client_id}")
        print(f"🔍 Искали в:")
        print(f"   📁 Документы: {client_docs_path}")
        print(f"   📁 FAISS индекс: {client_faiss_path}")
        return

    # Можно попробовать загрузить индекс клиента
    try:
        stats = processor.get_index_statistics()
    except Exception:
        stats = {}

    downloaded_files = list(client_docs_path.glob("**/*")) if client_docs_path.exists() else []
    faiss_files = list(client_faiss_path.glob("*")) if client_faiss_path.exists() else []

    print(f"📊 Что будет удалено для клиента {client_id}:")
    if stats.get('status') == 'ready':
        print(f"   📄 Документов в индексе: {stats.get('sources_count', 0)}")
        print(f"   🧩 Фрагментов в индексе: {stats.get('total_chunks', 0)}")

    print(f"   💾 Скачанных файлов: {len(downloaded_files)}")
    print(f"   🗄️ Файлов индекса: {len(faiss_files)}")

    if len(downloaded_files) == 0 and len(faiss_files) == 0 and stats.get('total_chunks', 0) == 0:
        print("✅ Данные уже удалены или отсутствуют!")
        return

    docs_size = sum(f.stat().st_size for f in downloaded_files if f.is_file()) / 1024 / 1024
    faiss_size = sum(f.stat().st_size for f in faiss_files if f.is_file()) / 1024 / 1024
    print(f"   💾 Освободится: ~{docs_size + faiss_size:.1f} MB")

    print(f"\n⚠️ ВНИМАНИЕ: Это удалит данные клиента {client_id}!")
    print(f"   - Все скачанные документы клиента")
    print(f"   - Весь FAISS индекс клиента")
    print(f"   - Все векторы и метаданные клиента")

    confirm = input(f"\nПродолжить? (y/N): ")

    if confirm.lower() == 'y':
        print(f"\n🗑️ Начинаем очистку данных клиента {client_id}...")

        # Сначала удаляем файлы напрямую, потом очищаем индекс
        removed_files = 0
        if client_docs_path.exists():
            try:
                shutil.rmtree(client_docs_path)
                removed_files = len(downloaded_files)
                print(f"✅ Удалено файлов документов: {removed_files}")
            except Exception as e:
                print(f"⚠️ Не удалось удалить документы: {e}")

        removed_index_files = 0
        if client_faiss_path.exists():
            try:
                shutil.rmtree(client_faiss_path)
                removed_index_files = len(faiss_files)
                print(f"✅ Удалено файлов индекса: {removed_index_files}")
            except Exception as e:
                print(f"⚠️ Не удалось удалить индекс: {e}")

        # Теперь очищаем индекс программно (если папка еще существует)
        try:
            if client_faiss_path.exists():
                processor.clear_all_data()
                print("✅ FAISS индекс очищен программно")
        except Exception as e:
            print(f"⚠️ Ошибка при программной очистке индекса: {e}")
            # Это не критично, так как файлы уже удалены

        print(f"\n🎉 Очистка данных клиента {client_id} завершена!")
        print(f"🔄 Система готова к новой загрузке документов для этого клиента.")

    else:
        print("❌ Очистка отменена")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python quick_cleanup.py <client_id>")
        sys.exit(1)
    client_id = sys.argv[1]

    try:
        quick_cleanup(client_id)
    except KeyboardInterrupt:
        print("\n👋 Очистка прервана")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")