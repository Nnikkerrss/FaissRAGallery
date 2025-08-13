from flask import Blueprint, request, jsonify
from .faiss_loader import load_documents_from_url
from .client_info_service import ClientInfoService
bp = Blueprint('faiss', __name__)

@bp.route("/faiss/search", methods=["POST"])
def search():
    data = request.json
    # Здесь твоя логика работы с FAISS
    return {"status": "ok", "query": data}

@bp.route("/faiss/index", methods=["GET"])
def index():
    return {"status": "ok faiss index"}

@bp.route("/faiss/create_index", methods=["POST"])
def create_index():
    try:
        data = request.get_json()
        clientId = data["client_id"]
        clientData = "https://1c.gwd.ru/services/hs/Request/GetData/GetAllFiles?api_key=5939a5cc-948b-4d78-98a7-370193831b70&client_id=" + clientId

        result = load_documents_from_url(clientData, log_level="INFO")

        if result['success']:
            return {
                "status": "ok",
                "clientId": clientId,
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
        clientId = data["client_id"]

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