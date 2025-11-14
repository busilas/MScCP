"""
Upload Routes

Handles document uploads for RAG indexing.
"""
import os
import hashlib
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.document import Document
from app.models.settings import Settings
from app.services.chatbot_service import chatbot_service
from app.services.guardian import guardian


upload_bp = Blueprint('upload', __name__)


ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_hash(filepath):
    """Calculate SHA256 hash of file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def extract_text_from_file(filepath, file_type):
    """
    Extract text content from uploaded file.

    Args:
        filepath: Path to file
        file_type: File extension (pdf, txt, csv)

    Returns:
        Extracted text content
    """
    if file_type == 'txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    elif file_type == 'csv':
        import csv
        content = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                content.append(' '.join(row))
        return '\n'.join(content)

    elif file_type == 'pdf':
        try:
            import PyPDF2
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = []
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
                return '\n'.join(text)
        except ImportError:
            return "PDF extraction requires PyPDF2. Install with: pip install PyPDF2"
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"

    return ""


@upload_bp.route('/')
@login_required
def index():
    """
    Document upload page.

    Shows upload form and user's uploaded documents.
    """
    user_documents = db.session.query(Document).filter_by(uploaded_by=current_user.id)\
        .order_by(Document.uploaded_at.desc())\
        .all()

    return render_template(
        'upload.html',
        documents=user_documents,
        allowed_extensions=ALLOWED_EXTENSIONS,
        max_size_mb=MAX_FILE_SIZE // (1024 * 1024)
    )


@upload_bp.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    """
    API endpoint for file upload.

    Validates file, extracts text, and adds to FAISS index.

    Request:
        multipart/form-data with 'file' field

    Response JSON:
        {
            "success": true,
            "document_id": 123,
            "filename": "document.pdf",
            "chunks_indexed": 15
        }
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Secure filename
        filename = secure_filename(file.filename)
        file_type = filename.rsplit('.', 1)[1].lower()

        # Create upload directory if needed
        upload_dir = os.path.join('uploads', str(current_user.id))
        os.makedirs(upload_dir, exist_ok=True)

        # Save file
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Get file size
        file_size = os.path.getsize(filepath)

        # Check file size
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB'}), 400

        # Calculate file hash
        file_hash = get_file_hash(filepath)

        # Check for duplicates
        existing_doc = db.session.query(Document).filter_by(content_hash=file_hash).first()
        if existing_doc:
            os.remove(filepath)
            return jsonify({'error': 'This document has already been uploaded'}), 400

        # Extract text
        text_content = extract_text_from_file(filepath, file_type)

        if not text_content or len(text_content) < 10:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from file or file is too short'}), 400

        # Scan for PII
        pii_scan = guardian.scan_document(text_content)
        if pii_scan['has_pii'] and pii_scan['total_matches'] > 10:
            os.remove(filepath)
            return jsonify({
                'error': 'Document contains significant PII and cannot be indexed',
                'pii_types': pii_scan['detected_types']
            }), 400

        # Create document record
        document = Document(
            filename=filename,
            filepath=filepath,
            file_type=file_type,
            file_size=file_size,
            uploaded_by=current_user.id,
            content_hash=file_hash
        )
        db.session.add(document)
        db.session.commit()

        # Add to FAISS index
        num_chunks = chatbot_service.add_document_to_rag(
            text_content,
            metadata={
                'doc_id': document.id,
                'filename': filename,
                'uploaded_by': current_user.id
            }
        )

        # Mark as indexed
        document.mark_indexed(num_chunks)

        return jsonify({
            'success': True,
            'document_id': document.id,
            'filename': filename,
            'chunks_indexed': num_chunks,
            'file_size': file_size
        }), 200

    except Exception as e:
        db.session.rollback()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@upload_bp.route('/api/documents/<int:doc_id>/index', methods=['POST'])
@login_required
def index_document(doc_id):
    """
    Index or re-index a document into RAG system.

    Args:
        doc_id: Document ID to index
    """
    document = db.session.get(Document, doc_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404

    # Check ownership
    if document.uploaded_by != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        # Check if file exists
        if not os.path.exists(document.filepath):
            return jsonify({'error': 'Document file not found on disk'}), 404

        # Extract text
        text_content = extract_text_from_file(document.filepath, document.file_type)

        if not text_content or len(text_content) < 10:
            return jsonify({'error': 'Could not extract text from file or file is too short'}), 400

        # Add to FAISS index
        num_chunks = chatbot_service.add_document_to_rag(
            text_content,
            metadata={
                'doc_id': document.id,
                'filename': document.filename,
                'uploaded_by': current_user.id
            }
        )

        # Mark as indexed
        document.mark_indexed(num_chunks)

        return jsonify({
            'success': True,
            'document_id': document.id,
            'filename': document.filename,
            'chunks_indexed': num_chunks
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Indexing failed: {str(e)}'}), 500


@upload_bp.route('/api/documents/<int:doc_id>', methods=['DELETE'])
@login_required
def delete_document(doc_id):
    """
    Delete uploaded document.

    Args:
        doc_id: Document ID to delete
    """
    document = db.session.get(Document, doc_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404

    # Check ownership
    if document.uploaded_by != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

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
        return jsonify({'error': f'Failed to delete: {str(e)}'}), 500
