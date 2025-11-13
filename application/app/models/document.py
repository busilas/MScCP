"""
Document Model

Manages user-uploaded documents for RAG retrieval.
"""
from datetime import datetime
from app import db


class Document(db.Model):
    """
    Document model for storing uploaded files metadata.

    Tracks documents uploaded by users for RAG-based retrieval.
    Stores file metadata and FAISS indexing status.

    Attributes:
        id: Primary key
        filename: Original filename
        filepath: Storage path on server
        file_type: MIME type (pdf, txt, csv)
        file_size: File size in bytes
        uploaded_by: Foreign key to User
        uploaded_at: Upload timestamp
        indexed: Whether document has been indexed in FAISS
        indexed_at: When document was indexed
        content_hash: SHA256 hash of content for deduplication
        num_chunks: Number of text chunks extracted
    """
    __tablename__ = 'documents'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    indexed = db.Column(db.Boolean, default=False, nullable=False)
    indexed_at = db.Column(db.DateTime, nullable=True)
    content_hash = db.Column(db.String(64), nullable=True, index=True)
    num_chunks = db.Column(db.Integer, default=0)

    def __repr__(self):
        """String representation of Document."""
        return f'<Document {self.filename}>'

    def to_dict(self):
        """
        Convert document to dictionary for JSON serialization.

        Returns:
            Dictionary representation of document
        """
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'uploaded_by': self.uploaded_by,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'indexed': self.indexed,
            'indexed_at': self.indexed_at.isoformat() if self.indexed_at else None,
            'num_chunks': self.num_chunks
        }

    def mark_indexed(self, num_chunks=0):
        """
        Mark document as indexed in FAISS.

        Args:
            num_chunks: Number of text chunks extracted
        """
        self.indexed = True
        self.indexed_at = datetime.utcnow()
        self.num_chunks = num_chunks
        db.session.commit()
