from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from .db import Base


class ModelRegistry(Base):
    """Registered ML models."""
    __tablename__ = "model_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(64), nullable=False)  # classification, regression, embedding, generation
    framework = Column(String(64), nullable=True)  # pytorch, tensorflow, onnx, sklearn
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ModelVersion(Base):
    """Versions of registered models."""
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    version = Column(String(64), nullable=False)
    location = Column(String(512), nullable=True)  # s3://, minio://, or local path
    file_size = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)  # accuracy, f1, etc.
    parameters = Column(JSON, nullable=True)  # hyperparameters
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TrainingJob(Base):
    """Training job records."""
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    name = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="pending")  # pending, running, completed, failed
    config = Column(JSON, nullable=True)  # training configuration
    dataset_location = Column(String(512), nullable=True)
    output_model_location = Column(String(512), nullable=True)
    metrics = Column(JSON, nullable=True)  # training metrics
    logs = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class InferenceJob(Base):
    """Inference job records."""
    __tablename__ = "inference_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    version_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class InferenceEndpoint(Base):
    """Deployed inference endpoints."""
    __tablename__ = "inference_endpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    version_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    endpoint_url = Column(String(512), nullable=True)
    status = Column(String(32), default="inactive")  # inactive, deploying, active, failed
    replicas = Column(Integer, default=1)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# ============================================
# OWNER-ONLY ML TRAINING TABLES
# ============================================

class OwnerVocabulary(Base):
    """Owner-managed vocabulary words for ML training."""
    __tablename__ = "owner_vocabulary"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    word = Column(String(255), nullable=False, unique=True, index=True)
    category = Column(String(64), nullable=True)
    frequency = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OwnerAnchor(Base):
    """Owner-managed anchors for semantic mapping."""
    __tablename__ = "owner_anchors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hash = Column(String(66), nullable=False, unique=True, index=True)  # 0x + 64 hex chars
    words = Column(JSON, nullable=False)  # List of 12 words
    xyz = Column(JSON, nullable=True)  # [x, y, z] coordinates
    resonance = Column(Float, nullable=True)
    proof = Column(JSON, nullable=True)  # merkle root and timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OwnerForbiddenWord(Base):
    """Owner-managed forbidden words excluded from training."""
    __tablename__ = "owner_forbidden_words"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    word = Column(String(255), nullable=False, unique=True, index=True)
    reason = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OwnerDataset(Base):
    """Owner-managed training datasets."""
    __tablename__ = "owner_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    words = Column(JSON, nullable=True)  # List of words in dataset
    anchors = Column(JSON, nullable=True)  # List of anchor references
    created_by = Column(String(255), nullable=True)  # Owner email
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
