from flask import Flask, jsonify
import importlib

app = Flask(__name__)

@app.route("/", strict_slashes=False)
def root():
    return jsonify({"status": "ok", "message": "Главная страница API"})

# Подключаем роуты
for module_name in ["faiss_vs.routes", "lk_assistant.routes"]:
    mod = importlib.import_module(module_name)
    app.register_blueprint(mod.bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
