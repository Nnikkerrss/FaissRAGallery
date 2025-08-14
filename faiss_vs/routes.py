import json

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import os
from pathlib import Path

from .faiss_loader import load_documents_from_url
from .client_info_service import ClientInfoService
from .src.document_processor import DocumentProcessor, create_multimodal_processor, \
    create_text_processor  # ✅ НОВЫЕ импорты

bp = Blueprint('faiss', __name__)


@bp.route("/faiss/search", methods=["POST"])
def search():
    """Универсальный поиск с поддержкой мультимодальности"""
    try:
        data = request.json
        client_id = data.get("client_id")
        query = data.get("query")

        if not client_id or not query:
            return jsonify({"error": "Требуются client_id и query"}), 400

        # ✅ ОБНОВЛЕНО: Создаем процессор с автоопределением режима
        processor = DocumentProcessor(client_id=client_id)

        # Параметры поиска
        k = data.get("k", 5)
        min_score = data.get("min_score", 0.0)
        search_mode = data.get("mode", "auto")  # ✅ НОВОЕ: режим поиска
        filters = data.get("filters")

        # Выполняем поиск
        results = processor.search_documents(
            query=query,
            k=k,
            min_score=min_score,
            filters=filters,
            search_mode=search_mode
        )

        # return jsonify({
        #     "status": "ok",
        #     "client_id": client_id,
        #     "query": query,
        #     "mode": search_mode,
        #     "visual_search_enabled": processor.enable_visual_search,  # ✅ НОВОЕ
        #     "results_count": len(results),
        #     "results": results
        # })
        result_data = {
            "status": "ok",
            "client_id": client_id,
            "query": query,
            "mode": search_mode,
            "visual_search_enabled": processor.enable_visual_search,  # ✅ НОВОЕ
            "results_count": len(results),
            "results": results
        }

        os.makedirs('logs', exist_ok=True)
        with open('logs/search_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        return jsonify(result_data)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/faiss/index", methods=["GET"])
def index():
    return {"status": "ok faiss index"}


@bp.route("/faiss/create_index", methods=["POST"])
def create_index():
    """Создание индекса с автоопределением режима"""
    try:
        data = request.get_json()
        client_id = data["client_id"]
        enable_visual = data.get("enable_visual_search")  # ✅ НОВОЕ: опциональный параметр

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


# ✅ НОВЫЕ endpoints для мультимодального поиска

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
        enable_visual = data.get("enable_visual_search", False)

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