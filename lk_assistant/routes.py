from flask import Blueprint, request, jsonify, render_template_string
from .assistant_manager import get_assistant_manager
import logging

# Создаем Blueprint для ассистента
bp = Blueprint('lk_assistant', __name__)

# Получаем менеджер ассистентов
assistant_manager = get_assistant_manager()


@bp.route('/assistant/index', methods=['GET'])
def index():
    """Информация о нейроассистенте"""
    return jsonify({
        "status": "ok",
        "service": "LK Assistant",
        "description": "Нейроассистент для личного кабинета на основе FAISS RAG",
        "version": "1.0.0",
        "cache_stats": assistant_manager.get_cache_stats(),
        "endpoints": {
            "/assistant/ask": "POST - Задать вопрос ассистенту",
            "/assistant/stats": "GET - Статистика клиента",
            "/assistant/suggestions": "GET - Предлагаемые вопросы",
            "/assistant/history": "GET - История разговора",
            "/assistant/search_category": "POST - Поиск в категории",
            "/assistant/categories": "GET - Список категорий",
            "/assistant/recent_documents": "GET - Недавние документы",
            "/assistant/health": "GET - Проверка здоровья",
            "/assistant/reload": "POST - Перезагрузка ассистента",
            "/assistant/manager_stats": "GET - Статистика менеджера",
            "/assistant/clear_cache": "POST - Очистка кэша",
            "/assistant/bulk_operations": "POST - Массовые операции"
        }
    })


@bp.route('/assistant/chat', methods=['GET'])
def chat_interface():
    """Веб-интерфейс для чата с ассистентом"""
    # Получаем client_id из параметров URL
    client_id = request.args.get('client_id', '6a2502fa-caaa-11e3-9af3-e41f13beb1d2')

    # HTML шаблон встроен прямо в код для простоты
    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Нейроассистент ЛК</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            height: calc(100vh - 40px);
            display: flex;
            flex-direction: column;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
            scroll-behavior: smooth;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: #3498db;
            color: white;
        }

        .message.assistant .message-avatar {
            background: #2ecc71;
            color: white;
        }

        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            line-height: 1.5;
            word-wrap: break-word;
        }

        .message.user .message-content {
            background: #3498db;
            color: white;
            border-bottom-right-radius: 5px;
        }

        .message.assistant .message-content {
            background: white;
            color: #2c3e50;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 5px;
        }

        .message-time {
            font-size: 12px;
            opacity: 0.7;
            margin-top: 5px;
        }

        .message-sources {
            margin-top: 10px;
            padding: 10px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 8px;
            font-size: 12px;
        }

        .source-link {
            color: #3498db;
            text-decoration: none;
            word-break: break-all;
        }

        .source-link:hover {
            text-decoration: underline;
        }

        .chat-input-area {
            background: white;
            border-top: 1px solid #e0e0e0;
            padding: 20px;
        }

        .input-group {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-container {
            flex: 1;
            position: relative;
        }

        #messageInput {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            resize: none;
            min-height: 50px;
            max-height: 120px;
            outline: none;
            transition: border-color 0.3s;
        }

        #messageInput:focus {
            border-color: #3498db;
        }

        #sendButton {
            background: #3498db;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            transition: background 0.3s;
        }

        #sendButton:hover:not(:disabled) {
            background: #2980b9;
        }

        #sendButton:disabled {
            background: #95a5a6;
            cursor: not-allowed;
        }

        .suggestions {
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .suggestion-chip {
            background: #e3f2fd;
            color: #1976d2;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            transition: background 0.3s;
        }

        .suggestion-chip:hover {
            background: #bbdefb;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 8px;
            margin-bottom: 20px;
        }

        .typing-indicator.show {
            display: flex;
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #95a5a6;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }

        .error-message {
            background: #fee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #c62828;
        }

        .welcome-message {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }

        .welcome-message h2 {
            margin-bottom: 10px;
            color: #2c3e50;
        }

        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            margin: 15px 0 10px 0;
            color: #2c3e50;
        }

        .markdown-content p {
            margin: 10px 0;
            line-height: 1.6;
        }

        .markdown-content strong {
            font-weight: 600;
        }

        .markdown-content a {
            color: #3498db;
            text-decoration: none;
            word-break: break-all;
        }

        .markdown-content a:hover {
            text-decoration: underline;
        }

        .markdown-content ul {
            margin: 10px 0;
            padding-left: 20px;
        }

        .markdown-content li {
            margin: 5px 0;
        }

        @media (max-width: 768px) {
            .container {
                height: 100vh;
                border-radius: 0;
                margin: 0;
            }

            body {
                padding: 0;
            }

            .message-content {
                max-width: 85%;
            }

            .header {
                padding: 15px 20px;
            }

            .header h1 {
                font-size: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🤖 Нейроассистент ЛК
            </h1>
            <div class="status" id="status">
                Подключение...
            </div>
        </div>

        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <h2>Добро пожаловать!</h2>
                    <p>Я ваш персональный ассистент. Могу помочь найти документы, ссылки и ответить на вопросы по вашим проектам.</p>
                </div>
            </div>

            <div class="chat-input-area">
                <div class="suggestions" id="suggestions">
                    <!-- Предложения будут загружены динамически -->
                </div>

                <div class="input-group">
                    <div class="input-container">
                        <textarea 
                            id="messageInput" 
                            placeholder="Задайте вопрос или попросите найти документ..."
                            rows="1"
                        ></textarea>
                    </div>
                    <button id="sendButton" title="Отправить сообщение">
                        ➤
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        class AssistantChat {
            constructor() {
                this.clientId = {{ client_id|tojson }};
                this.messagesContainer = document.getElementById('chatMessages');
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.statusElement = document.getElementById('status');
                this.suggestionsContainer = document.getElementById('suggestions');

                this.init();
            }

            init() {
                this.setupEventListeners();
                this.loadSuggestions();
                this.checkAssistantHealth();
            }

            setupEventListeners() {
                this.sendButton.addEventListener('click', () => this.sendMessage());

                this.messageInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });

                // Автоматическое изменение высоты textarea
                this.messageInput.addEventListener('input', (e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                });
            }

            async checkAssistantHealth() {
                try {
                    const response = await fetch(`/assistant/health?client_id=${this.clientId}`);
                    const data = await response.json();

                    if (data.success && data.health.is_ready) {
                        this.statusElement.textContent = '🟢 Готов к работе';
                        this.statusElement.style.background = 'rgba(46, 204, 113, 0.2)';
                    } else {
                        this.statusElement.textContent = '🟡 Нет данных';
                        this.statusElement.style.background = 'rgba(241, 196, 15, 0.2)';
                    }
                } catch (error) {
                    this.statusElement.textContent = '🔴 Ошибка';
                    this.statusElement.style.background = 'rgba(231, 76, 60, 0.2)';
                }
            }

            async loadSuggestions() {
                try {
                    const response = await fetch(`/assistant/suggestions?client_id=${this.clientId}`);
                    const data = await response.json();

                    if (data.success && data.suggestions) {
                        this.renderSuggestions(data.suggestions.slice(0, 4)); // Показываем первые 4
                    }
                } catch (error) {
                    console.error('Ошибка загрузки предложений:', error);
                }
            }

            renderSuggestions(suggestions) {
                this.suggestionsContainer.innerHTML = '';

                suggestions.forEach(suggestion => {
                    const chip = document.createElement('button');
                    chip.className = 'suggestion-chip';
                    chip.textContent = suggestion;
                    chip.addEventListener('click', () => {
                        this.messageInput.value = suggestion;
                        this.sendMessage();
                    });
                    this.suggestionsContainer.appendChild(chip);
                });
            }

            async sendMessage() {
                const message = this.messageInput.value.trim();
                if (!message || this.sendButton.disabled) return;

                // Добавляем сообщение пользователя
                this.addMessage(message, 'user');

                // Очищаем поле ввода и блокируем кнопку
                this.messageInput.value = '';
                this.messageInput.style.height = 'auto';
                this.setSendingState(true);

                // Показываем индикатор печати
                this.showTypingIndicator();

                try {
                    const response = await fetch('/assistant/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            client_id: this.clientId,
                            question: message,
                            context_limit: 3
                        })
                    });

                    const data = await response.json();
                    this.hideTypingIndicator();

                    if (data.success) {
                        this.addMessage(data.answer, 'assistant', data.sources);
                    } else {
                        this.addErrorMessage(data.error || 'Произошла ошибка при обработке запроса');
                    }

                } catch (error) {
                    this.hideTypingIndicator();
                    this.addErrorMessage('Ошибка сети. Проверьте подключение к интернету.');
                    console.error('Ошибка:', error);
                } finally {
                    this.setSendingState(false);
                }
            }

            addMessage(content, sender, sources = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;

                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.textContent = sender === 'user' ? '👤' : '🤖';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';

                // Обрабатываем markdown-подобный контент
                const formattedContent = this.formatMessage(content);
                contentDiv.innerHTML = formattedContent;

                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = new Date().toLocaleTimeString('ru-RU', {
                    hour: '2-digit',
                    minute: '2-digit'
                });

                const messageContentWrapper = document.createElement('div');
                messageContentWrapper.appendChild(contentDiv);
                messageContentWrapper.appendChild(timeDiv);

                // Добавляем источники для ответов ассистента
                if (sender === 'assistant' && sources && sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'message-sources';
                    sourcesDiv.innerHTML = '<strong>📚 Источники:</strong><br>' + 
                        sources.map(source => 
                            `• <a href="${source.url}" target="_blank" class="source-link">${source.file}</a> (${source.category})`
                        ).join('<br>');
                    messageContentWrapper.appendChild(sourcesDiv);
                }

                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContentWrapper);

                this.messagesContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }

            formatMessage(content) {
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // **bold**
        .replace(/🔗 (.*?): (https?:\/\/[^\s]+)/g, '🔗 <a href="$2" target="_blank" class="source-link">$1</a>') // Ссылки
        .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" class="source-link">$1</a>') // Обычные ссылки
}

            addErrorMessage(errorText) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = `❌ ${errorText}`;
                this.messagesContainer.appendChild(errorDiv);
                this.scrollToBottom();
            }

            showTypingIndicator() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator show';
                typingDiv.id = 'typingIndicator';

                typingDiv.innerHTML = `
                    <div class="message-avatar" style="background: #2ecc71; color: white;">🤖</div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span>Печатает</span>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                `;

                this.messagesContainer.appendChild(typingDiv);
                this.scrollToBottom();
            }

            hideTypingIndicator() {
                const typingIndicator = document.getElementById('typingIndicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }

            setSendingState(sending) {
                this.sendButton.disabled = sending;
                this.messageInput.disabled = sending;
                this.sendButton.textContent = sending ? '⏳' : '➤';
            }

            scrollToBottom() {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }
        }

        // Инициализируем чат когда страница загружена
        document.addEventListener('DOMContentLoaded', () => {
            new AssistantChat();
        });
    </script>
</body>
</html>
    '''

    return render_template_string(html_template, client_id=client_id)


@bp.route("/assistant/ask", methods=["POST"])
def ask():
    """
    Основной endpoint для вопросов к ассистенту

    Body: {
        "client_id": "uuid",
        "question": "текст вопроса",
        "context_limit": 3 (необязательно)
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Нет данных в запросе"
            }), 400

        client_id = data.get("client_id")
        question = data.get("question")
        context_limit = data.get("context_limit", 3)

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        if not question:
            return jsonify({
                "success": False,
                "error": "Не указан вопрос"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Задаем вопрос
        response = assistant.ask(question, context_limit=context_limit)

        # Добавляем информацию о клиенте
        response['client_id'] = client_id
        response['assistant_name'] = assistant.assistant_name

        return jsonify(response)

    except Exception as e:
        logging.error(f"Ошибка в /assistant/ask: {e}")
        return jsonify({
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        }), 500


@bp.route("/assistant/stats", methods=["GET"])
def get_stats():
    """
    Получение статистики клиента

    Params: client_id
    """
    try:
        client_id = request.args.get("client_id")

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Получаем статистику
        stats = assistant.get_client_stats()

        return jsonify({
            "success": True,
            "stats": stats
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/stats: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения статистики: {str(e)}"
        }), 500


@bp.route("/assistant/suggestions", methods=["GET"])
def get_suggestions():
    """
    Получение предлагаемых вопросов

    Params: client_id
    """
    try:
        client_id = request.args.get("client_id")

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Получаем предложения
        suggestions = assistant.suggest_questions()

        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "client_id": client_id
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/suggestions: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения предложений: {str(e)}"
        }), 500


@bp.route("/assistant/categories", methods=["GET"])
def get_categories():
    """
    Получение списка доступных категорий документов

    Params: client_id
    """
    try:
        client_id = request.args.get("client_id")

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Получаем категории
        categories = assistant.get_available_categories()

        return jsonify({
            "success": True,
            "categories": categories,
            "client_id": client_id,
            "total_categories": len(categories)
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/categories: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения категорий: {str(e)}"
        }), 500


@bp.route("/assistant/recent_documents", methods=["GET"])
def get_recent_documents():
    """
    Получение списка недавних документов

    Params: client_id, limit (необязательно, по умолчанию 5)
    """
    try:
        client_id = request.args.get("client_id")
        limit = request.args.get("limit", 5, type=int)

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Получаем недавние документы
        recent_docs = assistant.get_recent_documents(limit=limit)

        return jsonify({
            "success": True,
            "recent_documents": recent_docs,
            "client_id": client_id,
            "limit": limit
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/recent_documents: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения недавних документов: {str(e)}"
        }), 500


@bp.route("/assistant/history", methods=["GET"])
def get_history():
    """
    Получение истории разговора

    Params: client_id, limit (необязательно)
    """
    try:
        client_id = request.args.get("client_id")
        limit = request.args.get("limit", 10, type=int)

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Получаем историю
        history = assistant.get_conversation_history(limit=limit)

        return jsonify({
            "success": True,
            "history": history,
            "total_messages": len(assistant.conversation_history),
            "client_id": client_id
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/history: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения истории: {str(e)}"
        }), 500


@bp.route("/assistant/clear_history", methods=["POST"])
def clear_history():
    """
    Очистка истории разговора

    Body: {"client_id": "uuid"}
    """
    try:
        data = request.get_json()
        client_id = data.get("client_id") if data else None

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Очищаем историю
        assistant.clear_history()

        return jsonify({
            "success": True,
            "message": "История разговора очищена",
            "client_id": client_id
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/clear_history: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка очистки истории: {str(e)}"
        }), 500


@bp.route("/assistant/search_category", methods=["POST"])
def search_by_category():
    """
    Поиск в определенной категории документов

    Body: {
        "client_id": "uuid",
        "query": "поисковый запрос",
        "category": "название категории"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Нет данных в запросе"
            }), 400

        client_id = data.get("client_id")
        query = data.get("query")
        category = data.get("category")

        if not all([client_id, query, category]):
            return jsonify({
                "success": False,
                "error": "Не указаны обязательные поля: client_id, query, category"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Выполняем поиск
        result = assistant.search_by_category(query, category)

        # Добавляем информацию о клиенте
        result['client_id'] = client_id

        return jsonify(result)

    except Exception as e:
        logging.error(f"Ошибка в /assistant/search_category: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка поиска: {str(e)}"
        }), 500


@bp.route("/assistant/reload", methods=["POST"])
def reload_assistant():
    """
    Перезагрузка ассистента (полезно после обновления данных)

    Body: {"client_id": "uuid"}
    """
    try:
        data = request.get_json()
        client_id = data.get("client_id") if data else None

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Перезагружаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id, force_reload=True)

        return jsonify({
            "success": True,
            "message": "Ассистент перезагружен",
            "client_id": client_id,
            "is_ready": assistant.is_ready,
            "stats": assistant.get_client_stats()
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/reload: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка перезагрузки: {str(e)}"
        }), 500


@bp.route("/assistant/health", methods=["GET"])
def health_check():
    """
    Проверка здоровья ассистента для конкретного клиента

    Params: client_id
    """
    try:
        client_id = request.args.get("client_id")

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Не указан client_id"
            }), 400

        # Получаем ассистента через менеджер
        assistant = assistant_manager.get_assistant(client_id)

        # Проверяем статус
        health_status = {
            "client_id": client_id,
            "is_ready": assistant.is_ready,
            "assistant_name": assistant.assistant_name,
            "has_conversation_history": len(assistant.conversation_history) > 0,
            "service_status": "healthy"
        }

        if assistant.is_ready:
            stats = assistant.get_client_stats()
            health_status.update({
                "total_documents": stats.get('total_documents', 0),
                "total_chunks": stats.get('total_chunks', 0),
                "categories_count": len(stats.get('categories', []))
            })

        return jsonify({
            "success": True,
            "health": health_status
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/health: {e}")
        return jsonify({
            "success": False,
            "health": {
                "client_id": client_id,
                "service_status": "unhealthy",
                "error": str(e)
            }
        }), 500


@bp.route("/assistant/manager_stats", methods=["GET"])
def get_manager_stats():
    """
    Получение статистики менеджера ассистентов
    """
    try:
        cache_stats = assistant_manager.get_cache_stats()
        active_assistants = assistant_manager.list_active_assistants()

        return jsonify({
            "success": True,
            "manager_stats": cache_stats,
            "active_assistants": active_assistants,
            "active_count": len(active_assistants)
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/manager_stats: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения статистики менеджера: {str(e)}"
        }), 500


@bp.route("/assistant/clear_cache", methods=["POST"])
def clear_cache():
    """
    Очистка всего кэша ассистентов

    Body: {"confirm": true} (обязательно для безопасности)
    """
    try:
        data = request.get_json()

        if not data or not data.get("confirm"):
            return jsonify({
                "success": False,
                "error": "Для очистки кэша требуется подтверждение: {\"confirm\": true}"
            }), 400

        cleared_count = assistant_manager.clear_all_cache()

        return jsonify({
            "success": True,
            "message": f"Кэш очищен, удалено {cleared_count} ассистентов",
            "cleared_assistants": cleared_count
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/clear_cache: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка очистки кэша: {str(e)}"
        }), 500


@bp.route("/assistant/bulk_operations", methods=["POST"])
def bulk_operations():
    """
    Массовые операции с ассистентами

    Body: {
        "operation": "reload_all" | "get_all_stats" | "health_check_all",
        "client_ids": ["client1", "client2", ...] (опционально)
    }
    """
    try:
        data = request.get_json()

        if not data or not data.get("operation"):
            return jsonify({
                "success": False,
                "error": "Не указана операция"
            }), 400

        operation = data.get("operation")
        client_ids = data.get("client_ids", [])

        # Если client_ids не указаны, берем всех активных
        if not client_ids:
            active_assistants = assistant_manager.list_active_assistants()
            client_ids = list(active_assistants.keys())

        results = {}

        if operation == "reload_all":
            for client_id in client_ids:
                try:
                    assistant = assistant_manager.get_assistant(client_id, force_reload=True)
                    results[client_id] = {
                        "success": True,
                        "is_ready": assistant.is_ready
                    }
                except Exception as e:
                    results[client_id] = {
                        "success": False,
                        "error": str(e)
                    }

        elif operation == "get_all_stats":
            for client_id in client_ids:
                try:
                    assistant = assistant_manager.get_assistant(client_id)
                    results[client_id] = {
                        "success": True,
                        "stats": assistant.get_client_stats()
                    }
                except Exception as e:
                    results[client_id] = {
                        "success": False,
                        "error": str(e)
                    }

        elif operation == "health_check_all":
            for client_id in client_ids:
                try:
                    assistant_info = assistant_manager.get_assistant_info(client_id)
                    if assistant_info:
                        results[client_id] = {
                            "success": True,
                            "health": assistant_info,
                            "cached": True
                        }
                    else:
                        # Пробуем создать ассистента для проверки
                        assistant = assistant_manager.get_assistant(client_id)
                        results[client_id] = {
                            "success": True,
                            "health": {
                                "is_ready": assistant.is_ready,
                                "cached": False
                            }
                        }
                except Exception as e:
                    results[client_id] = {
                        "success": False,
                        "error": str(e)
                    }

        else:
            return jsonify({
                "success": False,
                "error": f"Неизвестная операция: {operation}"
            }), 400

        return jsonify({
            "success": True,
            "operation": operation,
            "processed_clients": len(client_ids),
            "results": results
        })

    except Exception as e:
        logging.error(f"Ошибка в /assistant/bulk_operations: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка массовой операции: {str(e)}"
        }), 500