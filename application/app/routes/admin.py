"""
Admin Routes

Admin dashboard for system configuration and analytics.
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from functools import wraps
from app import db
from app.models.settings import Settings
from app.models.log import Log
from app.models.document import Document
from app.models.training_file import TrainingFile
from app.models.user import User


admin_bp = Blueprint('admin', __name__)


def admin_required(f):
    """Decorator to ensure user is admin."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Admin access required.', 'error')
            return redirect(url_for('chat.index'))
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/')
@admin_required
def dashboard():
    """
    Admin dashboard page.

    Shows:
    - Current settings
    - System statistics
    - Recent activity
    """
    settings = Settings.get_settings()

    # Get statistics
    total_users = User.query.count()
    total_queries = Log.query.count()
    total_documents = Document.query.count()
    total_training_files = TrainingFile.query.count()

    # Recent logs
    recent_logs = Log.query.order_by(Log.timestamp.desc()).limit(10).all()

    return render_template(
        'admin.html',
        settings=settings,
        stats={
            'total_users': total_users,
            'total_queries': total_queries,
            'total_documents': total_documents,
            'total_training_files': total_training_files
        },
        recent_logs=recent_logs
    )


@admin_bp.route('/settings', methods=['POST'])
@admin_required
def update_settings():
    """
    Update system settings.

    Request JSON:
        {
            "auth_required": true/false,
            "chatbot_mode": "distilbert|llama|hybrid",
            "use_local_distilbert": true/false,
            "use_local_llama": true/false
        }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    settings = Settings.get_settings()

    try:
        # Update settings
        if 'auth_required' in data:
            settings.auth_required = bool(data['auth_required'])

        if 'chatbot_mode' in data:
            mode = data['chatbot_mode']
            if mode in ['distilbert', 'llama', 'hybrid']:
                settings.chatbot_mode = mode
            else:
                return jsonify({'error': 'Invalid chatbot mode'}), 400

        if 'use_local_distilbert' in data:
            settings.use_local_distilbert = bool(data['use_local_distilbert'])

        if 'use_local_llama' in data:
            settings.use_local_llama = bool(data['use_local_llama'])

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'settings': settings.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to update settings: {str(e)}'}), 500


@admin_bp.route('/performance')
@admin_required
def performance():
    """
    Performance analytics page.

    Shows:
    - Latency statistics by mode
    - Request counts
    - Guardian statistics
    - RAG performance
    """
    stats = Log.get_performance_stats()

    # Get documents and training files stats
    from app.services.chatbot_service import chatbot_service
    rag_stats = chatbot_service.get_rag_stats()
    training_stats = TrainingFile.get_dataset_stats()

    return render_template(
        'performance.html',
        stats=stats,
        rag_stats=rag_stats,
        training_stats=training_stats
    )


@admin_bp.route('/api/performance/data', methods=['GET'])
@admin_required
def performance_data():
    """
    API endpoint for performance data.

    Returns JSON with performance metrics for charts.
    """
    stats = Log.get_performance_stats()

    return jsonify(stats), 200


@admin_bp.route('/users')
@admin_required
def users():
    """
    User management page.

    List all users with management options.
    """
    all_users = User.query.order_by(User.created_at.desc()).all()

    return render_template(
        'users.html',
        users=all_users
    )


@admin_bp.route('/users/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    """
    Toggle user active status.

    Args:
        user_id: User ID to toggle
    """
    user = User.query.get_or_404(user_id)

    if user.id == current_user.id:
        return jsonify({'error': 'Cannot disable your own account'}), 400

    user.is_active = not user.is_active
    db.session.commit()

    return jsonify({
        'success': True,
        'user_id': user.id,
        'is_active': user.is_active
    }), 200


@admin_bp.route('/documents')
@admin_required
def documents():
    """
    Document management page.

    List all uploaded documents.
    """
    all_documents = Document.query.order_by(Document.uploaded_at.desc()).all()

    return render_template(
        'documents.html',
        documents=all_documents
    )


@admin_bp.route('/documents/<int:doc_id>/delete', methods=['POST'])
@admin_required
def delete_document(doc_id):
    """
    Delete a document.

    Args:
        doc_id: Document ID to delete
    """
    import os
    document = Document.query.get_or_404(doc_id)

    try:
        # Delete file from disk
        if os.path.exists(document.filepath):
            os.remove(document.filepath)

        # Delete from database
        db.session.delete(document)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Document deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to delete document: {str(e)}'}), 500


@admin_bp.route('/logs')
@admin_required
def logs():
    """
    System logs page.

    Displays query logs with filtering options.
    """
    page = request.args.get('page', 1, type=int)
    per_page = 50

    logs_query = Log.query.order_by(Log.timestamp.desc())

    # Filter by mode
    mode_filter = request.args.get('mode')
    if mode_filter:
        logs_query = logs_query.filter_by(mode_used=mode_filter)

    # Filter by guardian flag
    guardian_filter = request.args.get('guardian')
    if guardian_filter == 'true':
        logs_query = logs_query.filter_by(guardian_flag=True)

    # Paginate
    pagination = logs_query.paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'logs.html',
        logs=pagination.items,
        pagination=pagination
    )
