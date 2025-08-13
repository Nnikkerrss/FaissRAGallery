from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Optional, List
import uvicorn
import sys
from pathlib import Path
import json
from urllib.parse import quote

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.vectorstore.faiss_manager import FAISSManager

app = FastAPI(title="FAISS Chunks Viewer", description="Просмотр чанков из FAISS индекса")

# Глобальные переменные
faiss_manager = None
current_client_id = None


def initialize_faiss_manager(client_id: str):
    """Инициализирует FAISS manager для конкретного клиента"""
    global faiss_manager, current_client_id

    if faiss_manager is None or current_client_id != client_id:
        faiss_manager = FAISSManager(client_id=client_id)
        current_client_id = client_id
        loaded = faiss_manager.load_index()
        if not loaded:
            print(f"⚠️ Индекс для клиента {client_id} не найден или не загружен")
            return False
        else:
            print(f"✅ Индекс для клиента {client_id} загружен")
            return True
    return True


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с выбором клиента"""
    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FAISS Chunks Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .client-form { background: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            input[type="text"] { padding: 10px; width: 400px; border: 1px solid #ddd; border-radius: 4px; }
            button { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
            button:hover { background: #1976D2; }
            .example { font-size: 12px; color: #666; margin-top: 10px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 FAISS Chunks Viewer</h1>
            <p>Просмотр чанков и метаданных из FAISS индекса</p>

            <div class="client-form">
                <h3>Выберите клиента:</h3>
                <form method="GET" action="/chunks">
                    <input type="text" name="client_id" placeholder="Введите Client ID" required>
                    <button type="submit">Показать чанки</button>
                </form>
                <div class="example">
                    Пример: 6a2502fa-caaa-11e3-9af3-e41f13beb1d2
                </div>
            </div>

            <h3>Доступные endpoints:</h3>
            <ul>
                <li><strong>/chunks?client_id=XXX</strong> - просмотр всех чанков клиента</li>
                <li><strong>/chunks?client_id=XXX&source_file=filename.pdf</strong> - чанки конкретного файла</li>
                <li><strong>/stats?client_id=XXX</strong> - статистика индекса клиента</li>
                <li><strong>/search?client_id=XXX&query=текст</strong> - поиск по чанкам</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html


@app.get("/chunks", response_class=HTMLResponse)
async def view_chunks(
        client_id: str = Query(..., description="ID клиента"),
        source_file: Optional[str] = Query(None, description="Имя файла для фильтрации"),
        limit: int = Query(50, description="Максимум чанков для отображения")
):
    """Веб-страница с просмотром чанков"""

    # Инициализируем FAISS manager
    if not initialize_faiss_manager(client_id):
        raise HTTPException(status_code=404, detail=f"Индекс для клиента {client_id} не найден")

    # Получаем чанки
    if source_file:
        chunks = faiss_manager.get_chunks_by_source(source_file)
        title_suffix = f" для файла '{source_file}'"
    else:
        chunks = faiss_manager.get_all_chunks()
        title_suffix = ""

    # Ограничиваем количество для отображения
    chunks_limited = chunks[:limit] if len(chunks) > limit else chunks

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FAISS Chunks - {client_id}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ 
                max-width: 1400px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{ 
                background: linear-gradient(135deg, #2c3e50, #3498db); 
                color: white; 
                padding: 20px 30px;
                text-align: center;
            }}
            .controls {{ 
                padding: 20px 30px; 
                background: #f8f9fa; 
                border-bottom: 1px solid #dee2e6;
            }}
            .stats {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; 
                margin-bottom: 20px;
            }}
            .stat-card {{ 
                background: white; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-number {{ 
                font-size: 24px; 
                font-weight: bold; 
                color: #3498db;
            }}
            .stat-label {{ 
                font-size: 12px; 
                color: #666; 
                text-transform: uppercase;
            }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 0;
                background: white;
            }}
            th {{ 
                background: #3498db; 
                color: white; 
                padding: 15px 10px; 
                text-align: left; 
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            td {{ 
                padding: 12px 10px; 
                border-bottom: 1px solid #ecf0f1; 
                vertical-align: top;
            }}
            tr:hover {{ 
                background: #f8f9fa;
            }}
            .chunk-id {{ 
                font-family: 'Courier New', monospace; 
                background: #f1f2f6; 
                padding: 4px 8px; 
                border-radius: 4px; 
                font-size: 11px;
                word-break: break-all;
                max-width: 150px;
            }}
            .chunk-text {{ 
                max-width: 400px; 
                word-wrap: break-word; 
                line-height: 1.4;
                font-size: 13px;
            }}
            .metadata-container {{ 
                max-width: 350px; 
                max-height: 300px;
                overflow-y: auto;
                font-size: 11px;
            }}
            .metadata-item {{ 
                margin: 3px 0; 
                padding: 4px 6px; 
                background: #f8f9fa; 
                border-radius: 3px;
                border-left: 3px solid #3498db;
            }}
            .metadata-key {{ 
                font-weight: 600; 
                color: #2c3e50; 
                font-size: 10px;
                text-transform: uppercase;
                display: block;
            }}
            .metadata-value {{ 
                color: #34495e; 
                word-break: break-word;
                margin-top: 2px;
            }}
            .metadata-url {{ 
                color: #3498db; 
                text-decoration: none;
                word-break: break-all;
            }}
            .metadata-url:hover {{ 
                text-decoration: underline;
            }}
            .empty-value {{ 
                color: #95a5a6; 
                font-style: italic;
                font-size: 10px;
            }}
            .source-file {{ 
                background: #e8f5e8; 
                padding: 4px 8px; 
                border-radius: 4px;
                max-width: 200px;
                word-break: break-word;
                font-size: 12px;
            }}
            .filter-form {{ 
                display: flex; 
                gap: 10px; 
                align-items: center;
                flex-wrap: wrap;
            }}
            input, select {{ 
                padding: 8px; 
                border: 1px solid #ddd; 
                border-radius: 4px;
            }}
            button {{ 
                padding: 8px 16px; 
                background: #3498db; 
                color: white; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer;
            }}
            button:hover {{ 
                background: #2980b9;
            }}
            .pagination {{ 
                text-align: center; 
                padding: 20px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 FAISS Chunks Viewer</h1>
                <p>Клиент: {client_id}{title_suffix}</p>
            </div>

            <div class="controls">
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{len(chunks)}</div>
                        <div class="stat-label">Всего чанков</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(chunks_limited)}</div>
                        <div class="stat-label">Показано</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(set(chunk.get('source_file', '') for chunk in chunks))}</div>
                        <div class="stat-label">Уникальных файлов</div>
                    </div>
                </div>

                <form class="filter-form" method="GET">
                    <input type="hidden" name="client_id" value="{client_id}">
                    <input type="text" name="source_file" placeholder="Фильтр по файлу" value="{source_file or ''}">
                    <select name="limit">
                        <option value="25" {"selected" if limit == 25 else ""}>25 чанков</option>
                        <option value="50" {"selected" if limit == 50 else ""}>50 чанков</option>
                        <option value="100" {"selected" if limit == 100 else ""}>100 чанков</option>
                        <option value="500" {"selected" if limit == 500 else ""}>500 чанков</option>
                    </select>
                    <button type="submit">Применить</button>
                    <a href="/stats?client_id={client_id}" style="margin-left: 10px;">📈 Статистика</a>
                    <a href="/" style="margin-left: 10px;">🏠 Главная</a>
                </form>
            </div>

            <table>
                <thead>
                    <tr>
                        <th style="width: 120px;">ID чанка</th>
                        <th style="width: 150px;">Файл</th>
                        <th style="width: 60px;">Индекс</th>
                        <th style="width: 300px;">Текст</th>
                        <th style="width: 300px;">Метаданные</th>
                        <th style="width: 80px;">FAISS ID</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Добавляем строки таблицы
    for chunk in chunks_limited:
        # Превью текста
        text = chunk.get('text', '')
        preview_text = (text[:200] + "...") if len(text) > 200 else text
        preview_text = preview_text.replace('<', '&lt;').replace('>', '&gt;')

        # Обработка метаданных
        metadata = chunk.get('metadata', {})
        metadata_html = ""

        if metadata:
            # Важные поля показываем первыми
            important_fields = ['source_url', 'category', 'parent', 'date', 'guiddoc', 'object_id', 'title',
                                'description']

            # Сначала важные поля
            for field in important_fields:
                if field in metadata:
                    value = metadata[field]
                    if value:
                        # Русские названия
                        field_names = {
                            'source_url': 'Ссылка',
                            'category': 'Категория',
                            'parent': 'Родитель',
                            'date': 'Дата',
                            'guiddoc': 'GUID',
                            'object_id': 'ID объекта',
                            'title': 'Заголовок',
                            'description': 'Описание'
                        }

                        field_name = field_names.get(field, field)

                        # Специальное форматирование для ссылок
                        if field == 'source_url' and str(value).startswith('http'):
                            display_value = f'<a href="{value}" target="_blank" class="metadata-url">{str(value)[:50]}...</a>'
                        else:
                            display_value = str(value)[:100] + ('...' if len(str(value)) > 100 else '')

                        metadata_html += f'''
                        <div class="metadata-item">
                            <span class="metadata-key">{field_name}:</span>
                            <div class="metadata-value">{display_value}</div>
                        </div>
                        '''

            # Затем остальные поля
            for key, value in metadata.items():
                if key not in important_fields:
                    if value and str(value).strip():
                        display_value = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
                        metadata_html += f'''
                        <div class="metadata-item">
                            <span class="metadata-key">{key}:</span>
                            <div class="metadata-value">{display_value}</div>
                        </div>
                        '''
        else:
            metadata_html = '<div class="empty-value">Нет метаданных</div>'

        # Добавляем строку в таблицу
        html_content += f"""
        <tr>
            <td><div class="chunk-id">{chunk.get('chunk_id', 'N/A')[:20]}...</div></td>
            <td><div class="source-file">{chunk.get('source_file', 'N/A')}</div></td>
            <td style="text-align: center;">{chunk.get('chunk_index', 'N/A')}</td>
            <td><div class="chunk-text">{preview_text}</div></td>
            <td><div class="metadata-container">{metadata_html}</div></td>
            <td style="text-align: center;">{chunk.get('faiss_id', 'N/A')}</td>
        </tr>
        """

    # Закрываем таблицу и HTML
    html_content += """
                </tbody>
            </table>

            <div class="pagination">
    """

    if len(chunks) > limit:
        html_content += f"Показано {len(chunks_limited)} из {len(chunks)} чанков. Увеличьте лимит для просмотра всех."
    else:
        html_content += f"Показаны все {len(chunks)} чанков."

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


@app.get("/stats")
async def index_stats(client_id: str = Query(..., description="ID клиента")):
    """Возвращает статистику индекса в JSON"""
    if not initialize_faiss_manager(client_id):
        raise HTTPException(status_code=404, detail=f"Индекс для клиента {client_id} не найден")

    stats = faiss_manager.get_index_stats()

    # Добавляем дополнительную статистику
    if stats.get('status') == 'ready':
        chunks = faiss_manager.get_all_chunks()

        # Статистика по файлам
        files_stats = {}
        categories_stats = {}

        for chunk in chunks:
            source_file = chunk.get('source_file', 'unknown')
            files_stats[source_file] = files_stats.get(source_file, 0) + 1

            metadata = chunk.get('metadata', {})
            category = metadata.get('category', 'uncategorized')
            categories_stats[category] = categories_stats.get(category, 0) + 1

        stats['files_distribution'] = files_stats
        stats['categories_distribution'] = categories_stats
        stats['unique_files'] = len(files_stats)
        stats['unique_categories'] = len(categories_stats)

    return stats


@app.get("/search", response_class=HTMLResponse)
async def search_chunks(
        client_id: str = Query(..., description="ID клиента"),
        query: str = Query(..., description="Поисковый запрос"),
        k: int = Query(10, description="Количество результатов")
):
    """Поиск по чанкам"""
    if not initialize_faiss_manager(client_id):
        raise HTTPException(status_code=404, detail=f"Индекс для клиента {client_id} не найден")

    # Выполняем поиск
    results = faiss_manager.search(query, k=k)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Поиск - {query}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .result:hover {{ background: #f9f9f9; }}
            .score {{ background: #3498db; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
            .metadata {{ background: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 4px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Результаты поиска</h1>
            <p><strong>Запрос:</strong> "{query}" | <strong>Найдено:</strong> {len(results)} результатов</p>
            <a href="/chunks?client_id={client_id}">← Назад к чанкам</a>
            <hr>
    """

    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        category = metadata.get('category', 'Без категории')
        source_file = result.get('source_file', 'Неизвестный файл')
        score = result.get('score', 0)
        text = result.get('text', '')[:300] + ('...' if len(result.get('text', '')) > 300 else '')

        html_content += f"""
        <div class="result">
            <h3>{i}. {source_file} <span class="score">Score: {score:.3f}</span></h3>
            <p><strong>Категория:</strong> {category}</p>
            <p>{text}</p>
            <div class="metadata">
                <strong>Метаданные:</strong><br>
        """

        for key, value in metadata.items():
            if value and key in ['source_url', 'date', 'guiddoc', 'object_id']:
                html_content += f"{key}: {value}<br>"

        html_content += """
            </div>
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content


if __name__ == "__main__":
    print("🚀 Запуск FAISS Chunks Viewer")
    print("📁 Открыть: http://localhost:8000")
    print("📊 Статистика: http://localhost:8000/stats?client_id=YOUR_CLIENT_ID")
    print("🔍 Чанки: http://localhost:8000/chunks?client_id=YOUR_CLIENT_ID")

    uvicorn.run(app, host="0.0.0.0", port=8000)