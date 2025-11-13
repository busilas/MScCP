"""
Training Routes

Handles training data management and LLaMA fine-tuning.
"""
import os
import json
from flask import Blueprint, render_template, request, jsonify, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.training_file import TrainingFile
from functools import wraps


training_bp = Blueprint('training', __name__)


ALLOWED_TRAINING_EXTENSIONS = {'txt', 'json', 'jsonl', 'csv'}
TRAINING_FOLDER = 'training'


def admin_required(f):
    """Decorator to ensure user is admin."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function


def allowed_training_file(filename):
    """Check if training file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_TRAINING_EXTENSIONS


@training_bp.route('/')
@admin_required
def index():
    """
    Training data management page.

    Shows:
    - Uploaded training files
    - Dataset statistics
    - Training controls
    """
    training_files = TrainingFile.query.order_by(TrainingFile.uploaded_at.desc()).all()
    stats = TrainingFile.get_dataset_stats()

    return render_template(
        'training.html',
        training_files=training_files,
        stats=stats,
        allowed_extensions=ALLOWED_TRAINING_EXTENSIONS
    )


@training_bp.route('/upload', methods=['POST'])
@admin_required
def upload_training_file():
    """
    Upload training data file.

    Request:
        multipart/form-data with:
        - file: Training data file
        - file_type: 'raw', 'clean', or 'examples'
        - dataset_type: 'synthetic', 'forum', 'phrasebank', or null
        - description: Optional description

    Response JSON:
        {
            "success": true,
            "file_id": 123,
            "filename": "training.json"
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('file_type', 'raw')
    dataset_type = request.form.get('dataset_type')
    description = request.form.get('description', '')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_training_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_TRAINING_EXTENSIONS)}'}), 400

    if file_type not in ['raw', 'clean', 'examples']:
        return jsonify({'error': 'Invalid file type. Must be: raw, clean, or examples'}), 400

    try:
        # Secure filename
        filename = secure_filename(file.filename)

        # Determine folder
        if file_type == 'raw':
            folder = os.path.join(TRAINING_FOLDER, 'raw')
        elif file_type == 'clean':
            folder = os.path.join(TRAINING_FOLDER, 'clean')
        else:  # examples
            folder = os.path.join(TRAINING_FOLDER, 'examples')

        os.makedirs(folder, exist_ok=True)

        # Save file
        filepath = os.path.join(folder, filename)
        file.save(filepath)

        # Get file size
        file_size = os.path.getsize(filepath)

        # Count samples (for JSON/JSONL files)
        num_samples = 0
        file_ext = filename.rsplit('.', 1)[1].lower()

        if file_ext == 'json':
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        num_samples = len(data)
                    elif isinstance(data, dict):
                        num_samples = len(data.get('examples', []))
            except:
                pass

        elif file_ext == 'jsonl':
            try:
                with open(filepath, 'r') as f:
                    num_samples = sum(1 for line in f if line.strip())
            except:
                pass

        # Create training file record
        training_file = TrainingFile(
            filename=filename,
            filepath=filepath,
            file_type=file_type,
            dataset_type=dataset_type,
            num_samples=num_samples,
            uploaded_by=current_user.id,
            file_size=file_size,
            description=description
        )
        db.session.add(training_file)
        db.session.commit()

        return jsonify({
            'success': True,
            'file_id': training_file.id,
            'filename': filename,
            'num_samples': num_samples
        }), 200

    except Exception as e:
        db.session.rollback()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@training_bp.route('/files/<int:file_id>', methods=['DELETE'])
@admin_required
def delete_training_file(file_id):
    """
    Delete training file.

    Args:
        file_id: Training file ID to delete
    """
    training_file = TrainingFile.query.get_or_404(file_id)

    try:
        # Delete file from disk
        if os.path.exists(training_file.filepath):
            os.remove(training_file.filepath)

        # Delete from database
        db.session.delete(training_file)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Training file deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to delete: {str(e)}'}), 500


@training_bp.route('/files/<int:file_id>/download')
@admin_required
def download_training_file(file_id):
    """
    Download training file.

    Args:
        file_id: Training file ID to download
    """
    training_file = TrainingFile.query.get_or_404(file_id)

    if not os.path.exists(training_file.filepath):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        training_file.filepath,
        as_attachment=True,
        download_name=training_file.filename
    )


@training_bp.route('/preprocess', methods=['POST'])
@admin_required
def preprocess_data():
    """
    Preprocess raw training data.

    Request JSON:
        {
            "file_id": 123,
            "operations": ["lowercase", "remove_pii", "tokenize"]
        }

    Response JSON:
        {
            "success": true,
            "output_file": "cleaned_data.json",
            "num_samples": 500
        }
    """
    data = request.get_json()

    if not data or 'file_id' not in data:
        return jsonify({'error': 'No file_id provided'}), 400

    file_id = data['file_id']
    operations = data.get('operations', [])

    training_file = TrainingFile.query.get_or_404(file_id)

    try:
        # Import preprocessing utilities
        from utils.preprocess import preprocess_file

        # Preprocess file
        output_path, num_samples = preprocess_file(
            training_file.filepath,
            operations
        )

        # Create new training file record for cleaned data
        output_filename = os.path.basename(output_path)
        cleaned_file = TrainingFile(
            filename=output_filename,
            filepath=output_path,
            file_type='clean',
            dataset_type=training_file.dataset_type,
            num_samples=num_samples,
            uploaded_by=current_user.id,
            file_size=os.path.getsize(output_path),
            description=f'Preprocessed from {training_file.filename}',
            processed=True
        )
        db.session.add(cleaned_file)

        # Mark original as processed
        training_file.mark_processed()

        db.session.commit()

        return jsonify({
            'success': True,
            'output_file': output_filename,
            'num_samples': num_samples,
            'file_id': cleaned_file.id
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 500


@training_bp.route('/train', methods=['POST'])
@admin_required
def start_training():
    """
    Start LLaMA fine-tuning job.

    Request JSON:
        {
            "training_file_ids": [1, 2, 3],
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0002
        }

    Response JSON:
        {
            "success": true,
            "job_id": "train_123",
            "message": "Training started"
        }
    """
    data = request.get_json()

    if not data or 'training_file_ids' not in data:
        return jsonify({'error': 'No training files specified'}), 400

    training_file_ids = data['training_file_ids']
    epochs = data.get('epochs', 3)
    batch_size = data.get('batch_size', 4)
    learning_rate = data.get('learning_rate', 0.0002)

    try:
        # Get training files
        training_files = TrainingFile.query.filter(
            TrainingFile.id.in_(training_file_ids)
        ).all()

        if not training_files:
            return jsonify({'error': 'No valid training files found'}), 404

        # Start training (this would typically be a background job)
        from utils.train_llama import start_training_job

        job_id = start_training_job(
            files=[tf.filepath for tf in training_files],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Training started with {len(training_files)} files',
            'num_samples': sum(tf.num_samples for tf in training_files)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Training failed to start: {str(e)}'}), 500


@training_bp.route('/jobs/<job_id>/status')
@admin_required
def training_status(job_id):
    """
    Get training job status.

    Args:
        job_id: Training job ID

    Response JSON:
        {
            "status": "running|completed|failed",
            "progress": 0.75,
            "current_epoch": 2,
            "total_epochs": 3,
            "metrics": {...}
        }
    """
    try:
        from utils.train_llama import get_training_status

        status = get_training_status(job_id)
        return jsonify(status), 200

    except Exception as e:
        return jsonify({'error': f'Failed to get status: {str(e)}'}), 500
