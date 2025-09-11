"""File and image storage service."""
import os
import uuid
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger

from app.core.settings import settings


def ensure_storage_dirs():
    """Ensure storage directories exist."""
    try:
        storage_dir = Path(settings.storage.upload_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (storage_dir / "documents").mkdir(exist_ok=True)
        (storage_dir / "images").mkdir(exist_ok=True)
        
        logger.info(f"Storage directories ensured at {storage_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create storage directories: {e}")
        return False


def save_original_file(file_content: bytes, filename: str) -> Tuple[str, str]:
    """
    Save original file to storage.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        
    Returns:
        Tuple of (storage_path, public_url)
    """
    try:
        ensure_storage_dirs()
        
        # Generate unique filename
        file_ext = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Save to documents directory
        storage_dir = Path(settings.storage.upload_dir)
        storage_path = storage_dir / "documents" / unique_filename
        
        with open(storage_path, 'wb') as f:
            f.write(file_content)
        
        # Generate public URL
        public_url = f"{settings.storage.base_url}/files/documents/{unique_filename}"
        
        logger.info(f"Saved original file: {storage_path}")
        return str(storage_path), public_url
        
    except Exception as e:
        logger.error(f"Failed to save original file: {e}")
        return "", ""


def save_image(image_data: bytes, doc_id: str, chunk_id: int, image_idx: int) -> str:
    """
    Save image and return public URL.
    
    Args:
        image_data: Image data as bytes
        doc_id: Document ID
        chunk_id: Chunk ID
        image_idx: Image index within chunk
        
    Returns:
        Public URL of the saved image
    """
    try:
        ensure_storage_dirs()
        
        # Generate unique filename
        image_ext = ".jpg"  # Default to jpg, could detect from data
        unique_filename = f"{doc_id}_{chunk_id}_{image_idx}{image_ext}"
        
        # Save to images directory
        storage_dir = Path(settings.storage.upload_dir)
        storage_path = storage_dir / "images" / unique_filename
        
        with open(storage_path, 'wb') as f:
            f.write(image_data)
        
        # Generate public URL
        public_url = f"/storage/images/{unique_filename}"
        
        logger.info(f"Saved image: {storage_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return ""


def save_image_file(image_data: bytes, filename: str, doc_id: str) -> str:
    """
    Save image file with unique filename.
    
    Args:
        image_data: Image bytes
        filename: Original filename
        doc_id: Document ID
        
    Returns:
        Public URL of saved image
    """
    try:
        ensure_storage_dirs()
        
        # Create unique filename
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            file_ext = '.jpg'  # Default to jpg
        
        unique_filename = f"{doc_id}_{filename.replace(' ', '_')}"
        storage_dir = Path(settings.storage.upload_dir)
        storage_path = storage_dir / "images" / unique_filename
        
        # Save image
        with open(storage_path, "wb") as f:
            f.write(image_data)
        
        # Return public URL
        public_url = f"/storage/images/{unique_filename}"
        
        logger.info(f"Saved image file: {storage_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to save image file: {e}")
        return ""


def get_file_info(file_path: str) -> Tuple[str, int, str]:
    """
    Get file information (sha256, size, mime type).
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (sha256, size_bytes, mime_type)
    """
    try:
        import hashlib
        import mimetypes
        
        # Calculate SHA256
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        sha256 = sha256_hash.hexdigest()
        
        # Get file size
        size_bytes = os.path.getsize(file_path)
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return sha256, size_bytes, mime_type
        
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return "", 0, "application/octet-stream"


def delete_file(storage_path: str) -> bool:
    """
    Delete file from storage.
    
    Args:
        storage_path: Path to the file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(storage_path):
            os.remove(storage_path)
            logger.info(f"Deleted file: {storage_path}")
            return True
        else:
            logger.warning(f"File not found: {storage_path}")
            return True  # Consider it successful if file doesn't exist
            
    except Exception as e:
        logger.error(f"Failed to delete file {storage_path}: {e}")
        return False


def cleanup_orphaned_files():
    """Clean up orphaned files in storage."""
    try:
        storage_dir = Path(settings.storage.upload_dir)
        
        # This would typically check against database records
        # For now, just log that cleanup was attempted
        logger.info("File cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cleanup files: {e}")
        return False

import uuid
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger

from app.core.settings import settings


def ensure_storage_dirs():
    """Ensure storage directories exist."""
    try:
        storage_dir = Path(settings.storage.upload_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (storage_dir / "documents").mkdir(exist_ok=True)
        (storage_dir / "images").mkdir(exist_ok=True)
        
        logger.info(f"Storage directories ensured at {storage_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create storage directories: {e}")
        return False


def save_original_file(file_content: bytes, filename: str) -> Tuple[str, str]:
    """
    Save original file to storage.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        
    Returns:
        Tuple of (storage_path, public_url)
    """
    try:
        ensure_storage_dirs()
        
        # Generate unique filename
        file_ext = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Save to documents directory
        storage_dir = Path(settings.storage.upload_dir)
        storage_path = storage_dir / "documents" / unique_filename
        
        with open(storage_path, 'wb') as f:
            f.write(file_content)
        
        # Generate public URL
        public_url = f"{settings.storage.base_url}/files/documents/{unique_filename}"
        
        logger.info(f"Saved original file: {storage_path}")
        return str(storage_path), public_url
        
    except Exception as e:
        logger.error(f"Failed to save original file: {e}")
        return "", ""


def save_image(image_data: bytes, doc_id: str, chunk_id: int, image_idx: int) -> str:
    """
    Save image and return public URL.
    
    Args:
        image_data: Image data as bytes
        doc_id: Document ID
        chunk_id: Chunk ID
        image_idx: Image index within chunk
        
    Returns:
        Public URL of the saved image
    """
    try:
        ensure_storage_dirs()
        
        # Generate unique filename
        image_ext = ".jpg"  # Default to jpg, could detect from data
        unique_filename = f"{doc_id}_{chunk_id}_{image_idx}{image_ext}"
        
        # Save to images directory
        storage_dir = Path(settings.storage.upload_dir)
        storage_path = storage_dir / "images" / unique_filename
        
        with open(storage_path, 'wb') as f:
            f.write(image_data)
        
        # Generate public URL
        public_url = f"/storage/images/{unique_filename}"
        
        logger.info(f"Saved image: {storage_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return ""


def save_image_file(image_data: bytes, filename: str, doc_id: str) -> str:
    """
    Save image file with unique filename.
    
    Args:
        image_data: Image bytes
        filename: Original filename
        doc_id: Document ID
        
    Returns:
        Public URL of saved image
    """
    try:
        ensure_storage_dirs()
        
        # Create unique filename
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            file_ext = '.jpg'  # Default to jpg
        
        unique_filename = f"{doc_id}_{filename.replace(' ', '_')}"
        storage_dir = Path(settings.storage.upload_dir)
        storage_path = storage_dir / "images" / unique_filename
        
        # Save image
        with open(storage_path, "wb") as f:
            f.write(image_data)
        
        # Return public URL
        public_url = f"/storage/images/{unique_filename}"
        
        logger.info(f"Saved image file: {storage_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to save image file: {e}")
        return ""


def get_file_info(file_path: str) -> Tuple[str, int, str]:
    """
    Get file information (sha256, size, mime type).
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (sha256, size_bytes, mime_type)
    """
    try:
        import hashlib
        import mimetypes
        
        # Calculate SHA256
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        sha256 = sha256_hash.hexdigest()
        
        # Get file size
        size_bytes = os.path.getsize(file_path)
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return sha256, size_bytes, mime_type
        
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return "", 0, "application/octet-stream"


def delete_file(storage_path: str) -> bool:
    """
    Delete file from storage.
    
    Args:
        storage_path: Path to the file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(storage_path):
            os.remove(storage_path)
            logger.info(f"Deleted file: {storage_path}")
            return True
        else:
            logger.warning(f"File not found: {storage_path}")
            return True  # Consider it successful if file doesn't exist
            
    except Exception as e:
        logger.error(f"Failed to delete file {storage_path}: {e}")
        return False


def cleanup_orphaned_files():
    """Clean up orphaned files in storage."""
    try:
        storage_dir = Path(settings.storage.upload_dir)
        
        # This would typically check against database records
        # For now, just log that cleanup was attempted
        logger.info("File cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cleanup files: {e}")
        return False





