"""
Chat Routes

Handles chatbot interface and query processing.
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from app import db
from app.models.settings import Settings
from app.models.log import Log
from app.services.chatbot_service import chatbot_service
from functools import wraps


chat_bp = Blueprint('chat', __name__)


def auth_check(f):
    """
    Decorator to check if authentication is required.

    If auth is required and user is not logged in, return error.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        settings = Settings.get_settings()
        if settings.auth_required and not current_user.is_authenticated:
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return render_template('login_required.html'), 401
        return f(*args, **kwargs)
    return decorated_function


@chat_bp.route('/')
@auth_check
def index():
    """
    Main chat interface page.

    Display chatbot UI with current settings.
    """
    settings = Settings.get_settings()
    return render_template(
        'chat.html',
        settings=settings,
        user=current_user if current_user.is_authenticated else None
    )


@chat_bp.route('/api/query', methods=['POST'])
@auth_check
def query():
    """
    API endpoint for chatbot queries.

    Request JSON:
        {
            "query": "user question",
            "mode": "optional mode override"
        }

    Response JSON:
        {
            "response": "chatbot response",
            "mode_used": "distilbert|llama|hybrid",
            "latency_ms": 123,
            "intent": "detected intent",
            "entities": {...}
        }
    """
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    query_text = data['query'].strip()
    mode_override = data.get('mode')

    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        # Process query through chatbot service
        result = chatbot_service.process_query(
            query=query_text,
            mode=mode_override,
            user_id=current_user.id if current_user.is_authenticated else None
        )

        # Log query
        log_entry = Log(
            user_id=current_user.id if current_user.is_authenticated else None,
            query=query_text,
            response=result['response'],
            mode_used=result['mode_used'],
            latency_ms=result['latency_ms'],
            guardian_flag=result['guardian_flag'],
            similarity_score=result.get('similarity_score')
        )
        db.session.add(log_entry)
        db.session.commit()

        # Return response
        return jsonify({
            'response': result['response'],
            'mode_used': result['mode_used'],
            'latency_ms': result['latency_ms'],
            'intent': result.get('intent'),
            'entities': result.get('entities'),
            'metadata': result.get('metadata')
        }), 200

    except Exception as e:
        # Log error
        log_entry = Log(
            user_id=current_user.id if current_user.is_authenticated else None,
            query=query_text,
            mode_used='error',
            error=str(e)
        )
        db.session.add(log_entry)
        db.session.commit()

        return jsonify({'error': 'Failed to process query', 'details': str(e)}), 500


@chat_bp.route('/api/history', methods=['GET'])
@auth_check
def history():
    """
    Get chat history for current user.

    Query params:
        limit: Number of recent messages (default: 50)

    Response JSON:
        {
            "history": [
                {"query": "...", "response": "...", "timestamp": "..."},
                ...
            ]
        }
    """
    if not current_user.is_authenticated:
        return jsonify({'history': []}), 200

    limit = request.args.get('limit', 50, type=int)

    logs = db.session.query(Log).filter_by(user_id=current_user.id)\
        .order_by(Log.timestamp.desc())\
        .limit(limit)\
        .all()

    history = [
        {
            'query': log.query,
            'response': log.response,
            'mode': log.mode_used,
            'timestamp': log.timestamp.isoformat()
        }
        for log in reversed(logs)
    ]

    return jsonify({'history': history}), 200
