import json

from flask import Blueprint, request, jsonify
import tempfile
import os
import sys
import shutil
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from .faiss_loader import load_documents_from_url
from .client_info_service import ClientInfoService
from .src.document_processor import DocumentProcessor, create_multimodal_processor, create_text_processor  # ✅ НОВЫЕ импорты
from .src.config import settings
import requests


bp = Blueprint('faiss', __name__)
logger = logging.getLogger(__name__)

try:
    from .src.search.smart_search import SmartSearchEngine, SearchConfig
    SMART_SEARCH_AVAILABLE = True
    print("✅ Умный поиск доступен")
except ImportError as e:
    SMART_SEARCH_AVAILABLE = False
    print(f"⚠️ Умный поиск недоступен: {e}")

@bp.route("/faiss/index", methods=["GET"])
def index():
    return {"status": "ok faiss index"}


@bp.route("/faiss/search", methods=["POST"])
def search():
    """Поиск с поддержкой умного режима"""
    try:
        data = request.json
        client_id = data.get("client_id")
        object_id = data.get("object_id")
        query = data.get("query", "")
        mode = data.get("mode", "auto")  # auto, normal, smart
        k = data.get("k", 5)

        if not client_id or not query:
            return jsonify({
                "status": "error",
                "error": "Требуется client_id и query"
            }), 400

        # Создаем процессор
        processor = DocumentProcessor(client_id=client_id)

        # ✅ НОВАЯ ЛОГИКА: Выбираем режим поиска
        if mode == "smart" and SMART_SEARCH_AVAILABLE:
            # Умный поиск
            config = SearchConfig(
                min_score_threshold=data.get("min_score", 0.3),
                semantic_weight=data.get("semantic_weight", 0.7),
                keyword_weight=data.get("keyword_weight", 0.3)
            )

            smart_searcher = SmartSearchEngine(processor, config)
            results = smart_searcher.smart_search(query, k=k)

            if object_id:
                results = [r for r in results if str(r.get("metadata", {}).get("object_id")) == str(object_id)]

            # Форматируем результаты для умного поиска
            formatted_results = []
            for result in results:
                formatted_result = {
                    "chunk_id": result.get("chunk_id"),
                    "score": result.get("combined_score", result.get("score", 0)),
                    "search_type": "smart",
                    "text": result.get("text"),
                    "source_file": result.get("source_file"),
                    "metadata": result.get("metadata", {}),
                    "relevance_explanation": {
                        "original_score": result.get("original_score", 0),
                        "keyword_score": result.get("keyword_score", 0),
                        "intent_score": result.get("intent_score", 0),
                        "combined_score": result.get("combined_score", 0)
                    }
                }
                formatted_results.append(formatted_result)

            return jsonify({
                "status": "ok",
                "client_id": client_id,
                "query": query,
                "mode": "smart",
                "visual_search_enabled": getattr(processor, 'enable_visual_search', False),
                "results_count": len(formatted_results),
                "results": formatted_results
            })

        else:
            # Обычный поиск (как было)
            results = processor.search_documents(query, k=k)

            if object_id:
                results = [r for r in results if str(r.get("metadata", {}).get("object_id")) == str(object_id)]

            formatted_results = []
            for result in results:
                formatted_result = {
                    "chunk_id": result.get("chunk_id"),
                    "score": result.get("score", 0),
                    "search_type": "text",
                    "text": result.get("text"),
                    "source_file": result.get("source_file"),
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)

            return jsonify({
                "status": "ok",
                "client_id": client_id,
                "query": query,
                "mode": "normal",
                "visual_search_enabled": getattr(processor, 'enable_visual_search', False),
                "results_count": len(formatted_results),
                "results": formatted_results
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500



@bp.route("/faiss/create_index", methods=["POST"])
def create_index():
    """Создание индекса с автоопределением режима"""
    try:
        data = request.get_json()
        client_id = data["client_id"]
        enable_visual = True  # ✅ НОВОЕ: опциональный параметр

        clientData = "https://1c.gwd.ru/services/hs/Request/GetData/GetAllFiles?api_key=5939a5cc-948b-4d78-98a7-370193831b70&client_id=" + client_id

        result = load_documents_from_url(clientData, log_level="INFO")

        if result['success']:
            return {
                "status": "ok",
                "clientId": client_id,
                "visual_search_enabled": enable_visual,  # ✅ НОВОЕ
                "processed_documents": result['indexed'],
                "total_documents": result['total_documents'],
                "statistics": {
                    "downloaded": result['downloaded'],
                    "parsed": result['parsed'],
                    "chunked": result['chunked']
                }
            }
        return jsonify({
            "success": False,
            "error": result['error'],
            "error_type": "processing_error"
        }), 400

    except Exception as e:
        error_msg = f"Ошибка обработки документов: {str(e)}"
        return jsonify({
            "success": False,
            "error": error_msg,
            "error_type": "internal_error"
        }), 500


@bp.route("/faiss/get_index", methods=["GET"])
def get_index():
    try:
        data = request.get_json()
        client_id = data["client_id"]

        # Создаем сервис и получаем информацию
        service = ClientInfoService()
        result = service.get_client_info(client_id)

        return jsonify(result)
    except KeyError:
        return jsonify({
            'success': False,
            'error': 'Не указан client_id в запросе'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка при получении данных: {str(e)}'
        })

@bp.route("/faiss/search_multimodal", methods=["POST"])
def search_multimodal():
    """Мультимодальный поиск с поддержкой текста и изображений"""
    try:
        data = request.json
        client_id = data.get("client_id")

        if not client_id:
            return jsonify({"error": "Требуется client_id"}), 400

        # Создаем процессор с автоопределением режима
        processor = DocumentProcessor(client_id=client_id)

        # Параметры поиска
        text_query = data.get("query")
        search_mode = data.get("mode", "text")  # "text", "visual_description", "multimodal"
        k = data.get("k", 5)
        min_score = data.get("min_score", 0.0)

        # Выполняем поиск в зависимости от режима
        if search_mode == "text":
            results = processor.search_documents(
                query=text_query,
                k=k,
                min_score=min_score,
                search_mode="text"
            )

        elif search_mode == "visual_description" and processor.enable_visual_search:
            # Поиск изображений по текстовому описанию
            results = processor.search_by_text_description(
                text_description=text_query,
                k=k,
                search_images_only=True
            )

        else:
            return jsonify({
                "error": f"Неподдерживаемый режим поиска: {search_mode}",
                "available_modes": ["text", "visual_description"] if processor.enable_visual_search else ["text"]
            }), 400

        return jsonify({
            "success": True,
            "client_id": client_id,
            "query": text_query,
            "mode": search_mode,
            "visual_search_available": processor.enable_visual_search,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@bp.route("/faiss/search_similar_images", methods=["POST"])
def search_similar_images():
    """Поиск похожих изображений по загруженному изображению"""
    try:
        client_id = request.form.get("client_id")
        if not client_id:
            return jsonify({"error": "Требуется client_id"}), 400

        # Проверяем загруженное изображение
        if 'image' not in request.files:
            return jsonify({"error": "Необходимо загрузить изображение"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400

        # Параметры поиска
        k = int(request.form.get('k', 5))
        min_score = float(request.form.get('min_score', 0.0))

        # Создаем процессор
        processor = DocumentProcessor(client_id=client_id)

        if not processor.enable_visual_search:
            return jsonify({
                "error": "Мультимодальный поиск недоступен для этого клиента",
                "suggestion": "Создайте индекс с enable_visual_search=true"
            }), 503

        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = Path(temp_file.name)

        try:
            # Выполняем поиск похожих изображений
            results = processor.search_similar_images(
                query_image_path=temp_path,
                k=k,
                min_score=min_score
            )

            # Анализируем загруженное изображение
            analysis = processor.get_image_analysis(temp_path)

            return jsonify({
                "success": True,
                "client_id": client_id,
                "uploaded_image_analysis": analysis,
                "similar_images_count": len(results),
                "similar_images": results
            })

        finally:
            # Удаляем временный файл
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@bp.route("/faiss/search_by_description", methods=["POST"])
def search_by_description():
    """Поиск изображений по текстовому описанию"""
    try:
        data = request.json
        client_id = data.get("client_id")
        description = data.get("description")

        if not client_id or not description:
            return jsonify({"error": "Требуются client_id и description"}), 400

        # Создаем процессор
        processor = DocumentProcessor(client_id=client_id)

        if not processor.enable_visual_search:
            # Fallback на обычный текстовый поиск
            results = processor.search_documents(description, k=data.get("k", 5))
            search_mode = "text_fallback"
        else:
            # Визуальный поиск по описанию
            results = processor.search_by_text_description(
                text_description=description,
                k=data.get("k", 5),
                search_images_only=data.get("images_only", True)
            )
            search_mode = "visual_description"

        return jsonify({
            "success": True,
            "client_id": client_id,
            "description": description,
            "search_mode": search_mode,
            "visual_search_available": processor.enable_visual_search,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route("/faiss/analyze_image", methods=["POST"])
def analyze_image():
    """Анализ загруженного изображения с помощью CLIP"""
    try:
        client_id = request.form.get("client_id", "temp")

        if 'image' not in request.files:
            return jsonify({"error": "Необходимо загрузить изображение"}), 400

        file = request.files['image']

        # Создаем процессор
        try:
            processor = create_multimodal_processor(client_id)
        except Exception as e:
            return jsonify({
                "error": "Анализ изображений недоступен",
                "details": str(e),
                "suggestion": "Убедитесь, что установлены torch и CLIP"
            }), 503

        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = Path(temp_file.name)

        try:
            # Анализируем изображение
            analysis = processor.get_image_analysis(temp_path)

            return jsonify({
                "success": True,
                "filename": file.filename,
                "analysis": analysis
            })

        finally:
            # Удаляем временный файл
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route("/faiss/client_capabilities", methods=["GET"])
def get_client_capabilities():
    """Возвращает возможности поиска для клиента"""
    try:
        client_id = request.args.get("client_id")
        if not client_id:
            return jsonify({"error": "Требуется client_id"}), 400

        # Создаем процессор с автоопределением режима
        processor = DocumentProcessor(client_id=client_id)

        # Получаем информацию о режиме и статистику
        mode_info = processor.get_processing_mode_info()
        stats = processor.get_index_statistics()

        return jsonify({
            "success": True,
            "client_id": client_id,
            "processing_mode": mode_info,
            "capabilities": mode_info['capabilities'],
            "statistics": {
                'total_chunks': stats.get('total_chunks', 0),
                'text_vectors': stats.get('text_vectors_count', stats.get('total_vectors', 0)),
                'visual_vectors': stats.get('visual_vectors_count', 0),
                'images_count': stats.get('content_analysis', {}).get('images_count', 0),
                'documents_count': stats.get('content_analysis', {}).get('documents_count', 0),
                'visual_coverage': stats.get('content_analysis', {}).get('visual_coverage', 0)
            },
            "available_search_modes": [
                "text_search",
                "image_search_by_description",
                "similar_image_search",
                "multimodal_combined_search",
                "image_analysis"
            ] if mode_info['visual_search_enabled'] else ["text_search"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route("/faiss/create_multimodal_index", methods=["POST"])
def create_multimodal_index():
    """Создает новый индекс с явной поддержкой мультимодальности"""
    try:
        data = request.json
        client_id = data["client_id"]
        enable_visual = True

        # URL для загрузки данных
        client_data_url = f"https://1c.gwd.ru/services/hs/Request/GetData/GetAllFiles?api_key=5939a5cc-948b-4d78-98a7-370193831b70&client_id={client_id}"

        # Создаем процессор с нужным режимом
        if enable_visual:
            try:
                processor = create_multimodal_processor(client_id)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Не удалось создать мультимодальный процессор: {str(e)}",
                    "suggestion": "Убедитесь, что установлены torch и CLIP"
                }), 503
        else:
            processor = create_text_processor(client_id)

        # Загружаем и обрабатываем документы
        from .faiss_loader import DocumentLoader
        loader = DocumentLoader()

        # Загружаем JSON данные
        json_data, extracted_client_id = loader.download_json_data(client_data_url)
        json_path = loader.save_json_data(json_data, extracted_client_id)

        # Обрабатываем документы
        result = processor.process_documents_from_json(json_path)

        if result['success']:
            # Получаем статистику
            stats = processor.get_index_statistics()
            mode_info = processor.get_processing_mode_info()

            return jsonify({
                "status": "ok",
                "client_id": client_id,
                "multimodal_enabled": enable_visual,
                "processed_documents": result['indexed'],
                "statistics": {
                    "downloaded": result['downloaded'],
                    "parsed": result['parsed'],
                    "chunked": result['chunked'],
                    "images_processed": result.get('images_processed', 0),
                    "visual_vectors_created": result.get('visual_vectors_created', 0),
                    "text_vectors_created": result.get('text_vectors_created', 0)
                },
                "index_stats": stats,
                "capabilities": mode_info['capabilities']
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Неизвестная ошибка'),
                "details": result
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Ошибка создания индекса: {str(e)}"
        }), 500

@bp.route("/faiss/delete_client", methods=["POST"])
def delete_client():
    """
    Эндпоинт для удаления всех данных клиента
    Основан на логике quick_cleanup.py

    Принимает JSON: {"client_id": "xxx"}
    Возвращает статистику удаления
    """
    try:
        # Получаем данные из запроса
        data = request.get_json()
        if not data or 'client_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указан client_id в запросе',
                'error_type': 'missing_parameter'
            }), 400

        client_id = data['client_id']
        logger.info(f"🧹 Запрос на удаление данных клиента: {client_id}")

        # Создаем процессор для этого клиента
        processor = DocumentProcessor(client_id=client_id)

        # Определяем пути к данным клиента (как в quick_cleanup.py)
        client_docs_path = settings.CLIENTS_DIR / client_id / "documents"
        client_faiss_path = settings.FAISS_INDEX_DIR / "clients" / client_id

        # Проверяем существование данных
        if not client_docs_path.exists() and not client_faiss_path.exists():
            return jsonify({
                'success': False,
                'error': f'Нет данных для клиента {client_id}',
                'error_type': 'client_not_found',
                'searched_paths': {
                    'documents': str(client_docs_path),
                    'faiss_index': str(client_faiss_path)
                }
            }), 404

        # Собираем статистику ДО удаления
        stats_before = {}
        try:
            index_stats = processor.get_index_statistics()
            stats_before = {
                'documents_in_index': index_stats.get('sources_count', 0),
                'chunks_in_index': index_stats.get('total_chunks', 0),
                'vectors_in_index': index_stats.get('total_vectors', 0)
            }
        except Exception:
            # Если индекс поврежден или не загружается
            stats_before = {'index_error': 'Не удалось загрузить статистику индекса'}

        # Подсчитываем файлы для удаления
        downloaded_files = list(client_docs_path.glob("**/*")) if client_docs_path.exists() else []
        faiss_files = list(client_faiss_path.glob("*")) if client_faiss_path.exists() else []

        # Подсчитываем размер данных
        docs_size = sum(f.stat().st_size for f in downloaded_files if f.is_file()) / 1024 / 1024
        faiss_size = sum(f.stat().st_size for f in faiss_files if f.is_file()) / 1024 / 1024
        total_size_mb = docs_size + faiss_size

        deletion_stats = {
            'files_to_delete': len(downloaded_files),
            'index_files_to_delete': len(faiss_files),
            'estimated_size_mb': round(total_size_mb, 2)
        }

        logger.info(f"📊 Статистика перед удалением для {client_id}: {deletion_stats}")

        # Выполняем удаление
        removed_files = 0
        removed_index_files = 0
        errors = []

        # 1. Удаляем документы клиента
        if client_docs_path.exists():
            try:
                shutil.rmtree(client_docs_path)
                removed_files = len(downloaded_files)
                logger.info(f"✅ Удалены документы клиента: {removed_files} файлов")
            except Exception as e:
                error_msg = f"Ошибка при удалении документов: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # 2. Удаляем FAISS индекс клиента
        if client_faiss_path.exists():
            try:
                shutil.rmtree(client_faiss_path)
                removed_index_files = len(faiss_files)
                logger.info(f"✅ Удален FAISS индекс клиента: {removed_index_files} файлов")
            except Exception as e:
                error_msg = f"Ошибка при удалении индекса: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # 3. Дополнительная программная очистка (если что-то осталось)
        try:
            if not errors:  # Только если удаление прошло успешно
                processor.clear_all_data()
                logger.info("✅ Выполнена программная очистка индекса")
        except Exception as e:
            # Это не критично, так как файлы уже удалены
            logger.warning(f"⚠️ Программная очистка: {str(e)}")

        # Формируем ответ
        success = len(errors) == 0
        result = {
            'success': success,
            'client_id': client_id,
            'stats_before': stats_before,
            'deletion_summary': {
                'removed_document_files': removed_files,
                'removed_index_files': removed_index_files,
                'freed_space_mb': round(total_size_mb, 2)
            },
            'message': f"Данные клиента {client_id} {'успешно удалены' if success else 'частично удалены'}"
        }

        if errors:
            result['errors'] = errors
            result['warning'] = 'Удаление завершилось с ошибками'

        status_code = 200 if success else 207  # 207 = Multi-Status (частичный успех)

        logger.info(f"🎉 Удаление для {client_id} завершено: {'успешно' if success else 'с ошибками'}")

        return jsonify(result), status_code

    except Exception as e:
        error_msg = f"Критическая ошибка при удалении данных клиента: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_type': 'internal_error'
        }), 500

@bp.route("/faiss/update_client", methods=["POST"])
def update_client():
    """
    Обновляет клиента: вызывает /faiss/delete_client и /faiss/create_multimodal_index
    """
    try:
        data = request.get_json()
        client_id = data.get("client_id")
        enable_visual = True

        if not client_id:
            return jsonify({"success": False, "error": "client_id обязателен"}), 400

        base_url = "http://localhost:8000/faiss"  # ⚠️ смотри чтобы совпадало с твоим хостом/портом

        # 1. Удаление
        delete_resp = requests.post(f"{base_url}/delete_client", json={"client_id": client_id})
        delete_json = delete_resp.json()

        # 2. Создание нового индекса
        create_resp = requests.post(f"{base_url}/create_multimodal_index", json={
            "client_id": client_id,
            "enable_visual_search": enable_visual
        })
        create_json = create_resp.json()

        return jsonify({
            "success": True,
            "client_id": client_id,
            "delete_phase": delete_json,
            "create_phase": create_json,
            "message": f"Клиент {client_id} успешно обновлён"
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Ошибка при обновлении клиента: {e}"
        }), 500


@bp.route("/faiss/find_similar_to_existing", methods=["POST"])
def find_similar_to_existing():
    """Поиск изображений, похожих на уже существующее в индексе"""
    try:
        data = request.json
        client_id = data.get("client_id")
        source_file = data.get("source_file")
        k = data.get("k", 5)

        if not client_id or not source_file:
            return jsonify({"error": "Требуются client_id и source_file"}), 400

        # Создаем процессор
        processor = DocumentProcessor(client_id=client_id)

        if not processor.enable_visual_search:
            return jsonify({
                "error": "Визуальный поиск недоступен для этого клиента",
                "suggestion": "Создайте индекс с enable_visual_search=true"
            }), 503

        # Ищем похожие
        results = processor.get_similar_to_existing(source_file, k=k)

        return jsonify({
            "success": True,
            "client_id": client_id,
            "source_file": source_file,
            "similar_images_count": len(results),
            "similar_images": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route("/faiss/search_combined", methods=["POST"])
def search_combined():
    """Комбинированный мультимодальный поиск (текст + изображение)"""
    try:
        # Получаем данные из формы (для загрузки файла) и JSON
        client_id = request.form.get("client_id")
        text_query = request.form.get("text_query")
        text_weight = float(request.form.get("text_weight", 0.6))
        k = int(request.form.get("k", 5))

        if not client_id:
            return jsonify({"error": "Требуется client_id"}), 400

        # Создаем процессор
        processor = DocumentProcessor(client_id=client_id)

        if not processor.enable_visual_search:
            # Fallback на текстовый поиск
            if text_query:
                results = processor.search_documents(text_query, k=k)
                return jsonify({
                    "success": True,
                    "client_id": client_id,
                    "search_mode": "text_only_fallback",
                    "results": results
                })
            else:
                return jsonify({"error": "Мультимодальный поиск недоступен, требуется text_query"}), 400

        image_query_path = None

        # Обрабатываем изображение если есть
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                # Сохраняем временный файл
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    file.save(temp_file.name)
                    image_query_path = temp_file.name

        try:
            # Выполняем комбинированный поиск
            results = processor.search_multimodal(
                text_query=text_query,
                image_query_path=image_query_path,
                k=k,
                text_weight=text_weight
            )

            return jsonify({
                "success": True,
                "client_id": client_id,
                "text_query": text_query,
                "has_image_query": image_query_path is not None,
                "text_weight": text_weight,
                "search_mode": "multimodal_combined",
                "results_count": len(results),
                "results": results
            })

        finally:
            # Удаляем временный файл
            if image_query_path:
                Path(image_query_path).unlink(missing_ok=True)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route("/faiss/export_visual_vectors", methods=["POST"])
def export_visual_vectors():
    """Экспорт визуальных векторов клиента"""
    try:
        data = request.json
        client_id = data.get("client_id")

        if not client_id:
            return jsonify({"error": "Требуется client_id"}), 400

        # Создаем процессор
        processor = DocumentProcessor(client_id=client_id)

        if not processor.enable_visual_search:
            return jsonify({
                "error": "Визуальные векторы недоступны для этого клиента"
            }), 400

        # Экспортируем векторы
        export_data = processor.faiss_manager.export_visual_vectors()

        if 'error' in export_data:
            return jsonify({
                "success": False,
                "error": export_data['error']
            }), 400

        return jsonify({
            "success": True,
            "client_id": client_id,
            "export_data": export_data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Вспомогательные endpoint'ы для отладки и мониторинга

@bp.route("/faiss/health_check", methods=["GET"])
def health_check():
    """Проверка состояния системы мультимодального поиска"""
    try:
        from .src.data.image_processor import CLIP_AVAILABLE, OCR_AVAILABLE, check_dependencies

        dependencies = check_dependencies()

        return jsonify({
            "success": True,
            "system_status": "healthy",
            "dependencies": dependencies,
            "features_available": {
                "text_search": True,
                "visual_search": dependencies['clip_available'],
                "ocr_processing": dependencies['ocr_available'],
                "gpu_acceleration": dependencies['cuda_available']
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "system_status": "unhealthy",
            "error": str(e)
        }), 500

@bp.route("/faiss/materials", methods=["GET"])
def get_materials():
    """Получить список всех векторизованных материалов"""
    try:
        client_id = request.args.get('client_id')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        source_file = request.args.get('source_file')  # Фильтр по файлу
        category = request.args.get('category')  # Фильтр по категории

        if not client_id:
            return jsonify({
                "status": "error",
                "error": "Требуется client_id"
            }), 400

        processor = DocumentProcessor(client_id=client_id)

        # Получаем все чанки
        all_chunks = processor.faiss_manager.get_all_chunks()

        if not all_chunks:
            return jsonify({
                "status": "ok",
                "client_id": client_id,
                "total_chunks": 0,
                "materials": [],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_pages": 0
                }
            })

        # Применяем фильтры
        filtered_chunks = all_chunks

        if source_file:
            filtered_chunks = [chunk for chunk in filtered_chunks
                               if chunk.get('source_file', '').lower() == source_file.lower()]

        if category:
            filtered_chunks = [chunk for chunk in filtered_chunks
                               if chunk.get('metadata', {}).get('category', '').lower() == category.lower()]

        # Пагинация
        total_chunks = len(filtered_chunks)
        total_pages = (total_chunks + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_chunks = filtered_chunks[start_idx:end_idx]

        # Форматируем данные для ответа
        materials = []
        for chunk in paginated_chunks:
            metadata = chunk.get('metadata', {})
            material = {
                "chunk_id": chunk.get('chunk_id'),
                "source_file": chunk.get('source_file'),
                "chunk_index": metadata.get('chunk_index', 0),
                "text_preview": chunk.get('text', '')[:200] + ("..." if len(chunk.get('text', '')) > 200 else ''),
                "text_length": len(chunk.get('text', '')),
                "metadata": {
                    "title": metadata.get('title'),
                    "description": metadata.get('description'),
                    "category": metadata.get('category'),
                    "parent": metadata.get('parent'),
                    "date": metadata.get('date'),
                    "file_type": metadata.get('file_type'),
                    "file_size": metadata.get('file_size'),
                    "source_url": metadata.get('source_url'),
                    "processing_date": metadata.get('processing_date'),
                    # Дополнительные поля для изображений
                    "is_image": metadata.get('file_type', '').lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
                    "has_visual_embedding": metadata.get('has_visual_embedding', False),
                    "image_width": metadata.get('image_width'),
                    "image_height": metadata.get('image_height')
                }
            }
            materials.append(material)

        return jsonify({
            "status": "ok",
            "client_id": client_id,
            "total_chunks": total_chunks,
            "filtered_chunks": len(filtered_chunks),
            "materials": materials,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "filters": {
                "source_file": source_file,
                "category": category
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/faiss/materials/summary", methods=["GET"])
def get_materials_summary():
    """Получить сводку по материалам (статистика, категории, файлы)"""
    try:
        client_id = request.args.get('client_id')

        if not client_id:
            return jsonify({
                "status": "error",
                "error": "Требуется client_id"
            }), 400

        processor = DocumentProcessor(client_id=client_id)

        # Получаем статистику индекса
        stats = processor.get_index_statistics()

        if stats.get('status') != 'ready':
            return jsonify({
                "status": "ok",
                "client_id": client_id,
                "index_status": stats.get('status', 'not_ready'),
                "summary": None
            })

        all_chunks = processor.faiss_manager.get_all_chunks()

        # Анализируем файлы
        files_info = {}
        categories_info = {}
        file_types_info = {}

        for chunk in all_chunks:
            metadata = chunk.get('metadata', {})
            source_file = chunk.get('source_file', 'unknown')
            category = metadata.get('category', 'uncategorized')
            file_type = metadata.get('file_type', 'unknown')

            # Статистика по файлам
            if source_file not in files_info:
                files_info[source_file] = {
                    "chunks_count": 0,
                    "total_characters": 0,
                    "file_type": file_type,
                    "category": category,
                    "title": metadata.get('title', ''),
                    "description": metadata.get('description', ''),
                    "date": metadata.get('date', ''),
                    "file_size": metadata.get('file_size', 0),
                    "source_url": metadata.get('source_url', ''),
                    "is_image": file_type.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
                    "has_visual_embedding": metadata.get('has_visual_embedding', False)
                }

            files_info[source_file]["chunks_count"] += 1
            files_info[source_file]["total_characters"] += len(chunk.get('text', ''))

            # Статистика по категориям
            if category not in categories_info:
                categories_info[category] = {
                    "files_count": 0,
                    "chunks_count": 0,
                    "file_types": set()
                }

            if source_file not in [f["source_file"] for f in categories_info[category].get("files", [])]:
                categories_info[category]["files_count"] += 1

            categories_info[category]["chunks_count"] += 1
            categories_info[category]["file_types"].add(file_type)

            # Статистика по типам файлов
            if file_type not in file_types_info:
                file_types_info[file_type] = {
                    "files_count": 0,
                    "chunks_count": 0,
                    "total_size": 0
                }

            if source_file not in [f for f in files_info.keys() if files_info[f]["file_type"] == file_type]:
                file_types_info[file_type]["files_count"] += 1
                file_types_info[file_type]["total_size"] += metadata.get('file_size', 0)

            file_types_info[file_type]["chunks_count"] += 1

        # Конвертируем sets в lists для JSON
        for category in categories_info:
            categories_info[category]["file_types"] = list(categories_info[category]["file_types"])

        summary = {
            "overview": {
                "total_files": len(files_info),
                "total_chunks": len(all_chunks),
                "total_categories": len(categories_info),
                "total_file_types": len(file_types_info),
                "images_count": sum(1 for f in files_info.values() if f["is_image"]),
                "documents_count": sum(1 for f in files_info.values() if not f["is_image"]),
                "visual_embeddings_count": sum(1 for f in files_info.values() if f["has_visual_embedding"])
            },
            "files": files_info,
            "categories": categories_info,
            "file_types": file_types_info,
            "index_stats": stats
        }

        return jsonify({
            "status": "ok",
            "client_id": client_id,
            "index_status": "ready",
            "summary": summary
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/faiss/materials/file/<path:filename>", methods=["GET"])
def get_file_materials(filename):
    """Получить все чанки конкретного файла"""
    try:
        client_id = request.args.get('client_id')

        if not client_id:
            return jsonify({
                "status": "error",
                "error": "Требуется client_id"
            }), 400

        processor = DocumentProcessor(client_id=client_id)
        file_chunks = processor.faiss_manager.get_chunks_by_source(filename)

        if not file_chunks:
            return jsonify({
                "status": "ok",
                "client_id": client_id,
                "filename": filename,
                "chunks_count": 0,
                "chunks": [],
                "file_info": None
            })

        # Получаем информацию о файле из первого чанка
        first_chunk = file_chunks[0]
        metadata = first_chunk.get('metadata', {})

        file_info = {
            "filename": filename,
            "title": metadata.get('title'),
            "description": metadata.get('description'),
            "category": metadata.get('category'),
            "parent": metadata.get('parent'),
            "date": metadata.get('date'),
            "file_type": metadata.get('file_type'),
            "file_size": metadata.get('file_size'),
            "source_url": metadata.get('source_url'),
            "processing_date": metadata.get('processing_date'),
            "is_image": metadata.get('file_type', '').lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            "has_visual_embedding": metadata.get('has_visual_embedding', False),
            "total_chunks": len(file_chunks),
            "total_characters": sum(len(chunk.get('text', '')) for chunk in file_chunks)
        }

        # Форматируем чанки
        formatted_chunks = []
        for chunk in file_chunks:
            formatted_chunk = {
                "chunk_id": chunk.get('chunk_id'),
                "chunk_index": chunk.get('metadata', {}).get('chunk_index', 0),
                "text": chunk.get('text'),
                "text_length": len(chunk.get('text', '')),
                "start_char": chunk.get('metadata', {}).get('start_char', 0),
                "end_char": chunk.get('metadata', {}).get('end_char', 0),
                "chunk_size": chunk.get('metadata', {}).get('chunk_size', 0)
            }
            formatted_chunks.append(formatted_chunk)

        # Сортируем по индексу чанка
        formatted_chunks.sort(key=lambda x: x['chunk_index'])

        return jsonify({
            "status": "ok",
            "client_id": client_id,
            "filename": filename,
            "chunks_count": len(file_chunks),
            "file_info": file_info,
            "chunks": formatted_chunks
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/faiss/materials/categories", methods=["GET"])
def get_categories():
    """Получить список всех категорий с количеством материалов"""
    try:
        client_id = request.args.get('client_id')

        if not client_id:
            return jsonify({
                "status": "error",
                "error": "Требуется client_id"
            }), 400

        processor = DocumentProcessor(client_id=client_id)
        all_chunks = processor.faiss_manager.get_all_chunks()

        categories = {}
        files_by_category = {}

        for chunk in all_chunks:
            metadata = chunk.get('metadata', {})
            category = metadata.get('category', 'uncategorized')
            source_file = chunk.get('source_file', 'unknown')

            if category not in categories:
                categories[category] = {
                    "chunks_count": 0,
                    "files": set(),
                    "file_types": set()
                }
                files_by_category[category] = set()

            categories[category]["chunks_count"] += 1
            categories[category]["files"].add(source_file)
            categories[category]["file_types"].add(metadata.get('file_type', 'unknown'))
            files_by_category[category].add(source_file)

        # Конвертируем sets в lists и добавляем counts
        formatted_categories = []
        for category, info in categories.items():
            formatted_categories.append({
                "category": category,
                "chunks_count": info["chunks_count"],
                "files_count": len(info["files"]),
                "files": list(info["files"]),
                "file_types": list(info["file_types"])
            })

        # Сортируем по количеству файлов
        formatted_categories.sort(key=lambda x: x['files_count'], reverse=True)

        return jsonify({
            "status": "ok",
            "client_id": client_id,
            "total_categories": len(formatted_categories),
            "categories": formatted_categories
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/faiss/materials/delete", methods=["DELETE"])
def delete_material():
    """Удалить материал из индекса"""
    try:
        data = request.json
        client_id = data.get('client_id')
        source_file = data.get('source_file')
        chunk_id = data.get('chunk_id')  # Опционально - удалить конкретный чанк

        if not client_id:
            return jsonify({
                "status": "error",
                "error": "Требуется client_id"
            }), 400

        if not source_file and not chunk_id:
            return jsonify({
                "status": "error",
                "error": "Требуется source_file или chunk_id"
            }), 400

        processor = DocumentProcessor(client_id=client_id)

        if chunk_id:
            # Удаляем конкретный чанк
            success = processor.faiss_manager.remove_chunks([chunk_id])
            if success:
                processor.faiss_manager.save_index()
                return jsonify({
                    "status": "ok",
                    "message": f"Чанк {chunk_id} удален",
                    "deleted_chunks": 1
                })
            else:
                return jsonify({
                    "status": "error",
                    "error": "Не удалось удалить чанк"
                }), 500

        elif source_file:
            # Удаляем весь файл
            success = processor.remove_document(source_file)
            if success:
                return jsonify({
                    "status": "ok",
                    "message": f"Файл {source_file} удален из индекса"
                })
            else:
                return jsonify({
                    "status": "error",
                    "error": "Файл не найден или не удалось удалить"
                }), 404

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/faiss/system_info", methods=["GET"])
def system_info():
    """Информация о системе и доступных возможностях"""
    try:
        from .src.data.image_processor import check_dependencies
        from .src.config import settings

        dependencies = check_dependencies()

        return jsonify({
            "success": True,
            "system_info": {
                "text_model": settings.EMBEDDING_MODEL,
                "visual_model": settings.CLIP_MODEL if dependencies['clip_available'] else None,
                "text_dimension": settings.EMBEDDING_DIMENSION,
                "visual_dimension": settings.VISUAL_EMBEDDING_DIMENSION,
                "device": settings.get_device_for_processing(),
                "keep_files": settings.KEEP_DOWNLOADED_FILES,
                "visual_search_default": settings.ENABLE_VISUAL_SEARCH_BY_DEFAULT
            },
            "dependencies": dependencies,
            "supported_features": {
                "text_search": True,
                "visual_search": dependencies['clip_available'],
                "multimodal_search": dependencies['clip_available'],
                "image_analysis": dependencies['clip_available'],
                "ocr_processing": dependencies['ocr_available']
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500