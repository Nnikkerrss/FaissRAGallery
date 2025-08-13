from flask import Blueprint, request

bp = Blueprint('lk_assistant', __name__)

@bp.route('/lk_assistant/index', methods=['GET'])
def index():
    return {"status": "ok lk assistant",}
    # return render_template("lk_assistant/index.html")

@bp.route("/assistant/ask", methods=["POST"])
def ask():
    data = request.json
    # Здесь логика ассистента
    return {"status": "ok", "message": data}