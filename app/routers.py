from typing import Any, Dict, List, Optional
from datetime import datetime
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .models import (
    ModelRegistry, ModelVersion, TrainingJob, 
    InferenceJob, InferenceEndpoint
)


router = APIRouter(prefix="/ml", tags=["ml"])


# Request/Response Models
class ModelCreate(BaseModel):
    name: str
    model_type: str  # classification, regression, embedding, generation
    framework: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    model_type: str
    framework: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]
    is_active: bool

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    """Compatibility response for prediction endpoints.

    This wraps InferenceJob records into a simpler shape that the
    frontend's Prediction type can consume.
    """

    id: str
    created_at: datetime
    status: str
    model_id: str
    output: Optional[Dict[str, Any]] = None


class PredictionListResponse(BaseModel):
    items: List[PredictionResponse]
    total: int
    page: int
    limit: int


class VersionCreate(BaseModel):
    version: str
    location: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    is_default: bool = False


class VersionResponse(BaseModel):
    id: str
    model_id: str
    version: str
    location: Optional[str]
    metrics: Optional[Dict[str, Any]]
    is_default: bool

    class Config:
        from_attributes = True


class TrainingJobCreate(BaseModel):
    name: Optional[str] = None
    model_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    dataset_location: Optional[str] = None


class TrainingJobResponse(BaseModel):
    id: str
    name: Optional[str]
    status: str
    metrics: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class InferenceRequest(BaseModel):
    model_id: str
    version_id: Optional[str] = None
    input_data: Dict[str, Any]


class InferenceResponse(BaseModel):
    id: str
    model_id: str
    status: str
    output_data: Optional[Dict[str, Any]]
    latency_ms: Optional[int]
    error_message: Optional[str]

    class Config:
        from_attributes = True


# Model Registry Endpoints
@router.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def register_model(
    payload: ModelCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Register a new ML model."""
    user_id = request.headers.get("x-user-id")

    model = ModelRegistry(
        user_id=user_id,
        name=payload.name,
        model_type=payload.model_type,
        framework=payload.framework,
        description=payload.description,
        tags=payload.tags,
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)

    return ModelResponse(
        id=str(model.id),
        name=model.name,
        model_type=model.model_type,
        framework=model.framework,
        description=model.description,
        tags=model.tags,
        is_active=model.is_active,
    )


@router.get("/models", response_model=List[ModelResponse])
async def list_models(
    model_type: Optional[str] = None,
    framework: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """List all registered models."""
    stmt = select(ModelRegistry).where(ModelRegistry.is_active == True)
    if model_type:
        stmt = stmt.where(ModelRegistry.model_type == model_type)
    if framework:
        stmt = stmt.where(ModelRegistry.framework == framework)

    result = await session.execute(stmt)
    models = result.scalars().all()

    return [
        ModelResponse(
            id=str(m.id),
            name=m.name,
            model_type=m.model_type,
            framework=m.framework,
            description=m.description,
            tags=m.tags,
            is_active=m.is_active,
        )
        for m in models
    ]


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a model by ID."""
    result = await session.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelResponse(
        id=str(model.id),
        name=model.name,
        model_type=model.model_type,
        framework=model.framework,
        description=model.description,
        tags=model.tags,
        is_active=model.is_active,
    )


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Soft delete a model."""
    result = await session.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model.is_active = False
    await session.commit()
    return {"status": "deleted", "id": model_id}


# Model Version Endpoints
@router.post("/models/{model_id}/versions", response_model=VersionResponse, status_code=status.HTTP_201_CREATED)
async def create_version(
    model_id: str,
    payload: VersionCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new model version."""
    # Verify model exists
    result = await session.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # If this is default, unset other defaults
    if payload.is_default:
        stmt = select(ModelVersion).where(
            ModelVersion.model_id == model_id,
            ModelVersion.is_default == True,
        )
        result = await session.execute(stmt)
        for v in result.scalars().all():
            v.is_default = False

    version = ModelVersion(
        model_id=model_id,
        version=payload.version,
        location=payload.location,
        metrics=payload.metrics,
        parameters=payload.parameters,
        is_default=payload.is_default,
    )
    session.add(version)
    await session.commit()
    await session.refresh(version)

    return VersionResponse(
        id=str(version.id),
        model_id=str(version.model_id),
        version=version.version,
        location=version.location,
        metrics=version.metrics,
        is_default=version.is_default,
    )


@router.get("/models/{model_id}/versions", response_model=List[VersionResponse])
async def list_versions(
    model_id: str,
    session: AsyncSession = Depends(get_session),
):
    """List all versions of a model."""
    result = await session.execute(
        select(ModelVersion)
        .where(ModelVersion.model_id == model_id)
        .order_by(ModelVersion.created_at.desc())
    )
    versions = result.scalars().all()

    return [
        VersionResponse(
            id=str(v.id),
            model_id=str(v.model_id),
            version=v.version,
            location=v.location,
            metrics=v.metrics,
            is_default=v.is_default,
        )
        for v in versions
    ]


# Training Job Endpoints
@router.post("/training", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    payload: TrainingJobCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a new training job."""
    user_id = request.headers.get("x-user-id")

    job = TrainingJob(
        model_id=payload.model_id,
        user_id=user_id,
        name=payload.name,
        config=payload.config,
        dataset_location=payload.dataset_location,
        status="pending",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # In production, this would queue the job for async execution
    # For now, simulate immediate completion
    job.status = "completed"
    job.started_at = datetime.utcnow()
    job.completed_at = datetime.utcnow()
    job.metrics = {"accuracy": 0.95, "loss": 0.05}  # Placeholder
    await session.commit()

    return TrainingJobResponse(
        id=str(job.id),
        name=job.name,
        status=job.status,
        metrics=job.metrics,
    )


@router.get("/training", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List training jobs."""
    stmt = select(TrainingJob).order_by(TrainingJob.created_at.desc())
    if status:
        stmt = stmt.where(TrainingJob.status == status)

    result = await session.execute(stmt.limit(limit))
    jobs = result.scalars().all()

    return [
        TrainingJobResponse(
            id=str(j.id),
            name=j.name,
            status=j.status,
            metrics=j.metrics,
        )
        for j in jobs
    ]


@router.get("/training/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a training job by ID."""
    result = await session.execute(
        select(TrainingJob).where(TrainingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingJobResponse(
        id=str(job.id),
        name=job.name,
        status=job.status,
        metrics=job.metrics,
    )


# Inference Endpoints
@router.post("/infer", response_model=InferenceResponse)
async def run_inference(
    payload: InferenceRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Run inference on a model."""
    user_id = request.headers.get("x-user-id")
    start_time = time.time()

    # Verify model exists
    result = await session.execute(
        select(ModelRegistry).where(ModelRegistry.id == payload.model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get version (default if not specified)
    version_id = payload.version_id
    if not version_id:
        result = await session.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == payload.model_id,
                ModelVersion.is_default == True,
            )
        )
        version = result.scalar_one_or_none()
        if version:
            version_id = str(version.id)

    # Create inference job
    job = InferenceJob(
        model_id=payload.model_id,
        version_id=version_id,
        user_id=user_id,
        status="running",
        input_data=payload.input_data,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    try:
        # Placeholder inference logic
        # In production, this would call the actual model
        output_data = {
            "prediction": "placeholder_result",
            "confidence": 0.95,
            "model_type": model.model_type,
        }

        latency_ms = int((time.time() - start_time) * 1000)

        job.status = "completed"
        job.output_data = output_data
        job.latency_ms = latency_ms
        await session.commit()

        return InferenceResponse(
            id=str(job.id),
            model_id=str(job.model_id),
            status=job.status,
            output_data=job.output_data,
            latency_ms=job.latency_ms,
            error_message=None,
        )

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.latency_ms = int((time.time() - start_time) * 1000)
        await session.commit()

        return InferenceResponse(
            id=str(job.id),
            model_id=str(job.model_id),
            status=job.status,
            output_data=None,
            latency_ms=job.latency_ms,
            error_message=job.error_message,
        )


@router.get("/inference/{job_id}", response_model=InferenceResponse)
async def get_inference_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get an inference job by ID."""
    result = await session.execute(
        select(InferenceJob).where(InferenceJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    return InferenceResponse(
        id=str(job.id),
        model_id=str(job.model_id),
        status=job.status,
        output_data=job.output_data,
        latency_ms=job.latency_ms,
        error_message=job.error_message,
    )


# ============== Predictions Compatibility Endpoints ==============

@router.get("/predictions", response_model=PredictionListResponse)
async def list_predictions(
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List inference jobs as predictions (compat layer for old frontend).

    Results are ordered by created_at DESC.
    """

    if page < 1:
        page = 1
    if limit <= 0 or limit > 100:
        limit = 50

    # Total count
    total_result = await session.execute(select(func.count()).select_from(InferenceJob))
    total = total_result.scalar_one() or 0

    # Page of jobs
    stmt = (
        select(InferenceJob)
        .order_by(InferenceJob.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    result = await session.execute(stmt)
    jobs = result.scalars().all()

    items = [
        PredictionResponse(
            id=str(j.id),
            created_at=j.created_at,
            status=j.status,
            model_id=str(j.model_id),
            output=j.output_data,
        )
        for j in jobs
    ]

    return PredictionListResponse(items=items, total=total, page=page, limit=limit)


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a single prediction by ID (backed by InferenceJob)."""

    result = await session.execute(
        select(InferenceJob).where(InferenceJob.id == prediction_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return PredictionResponse(
        id=str(job.id),
        created_at=job.created_at,
        status=job.status,
        model_id=str(job.model_id),
        output=job.output_data,
    )


@router.get("/health")
async def health():
    return {"service": "ml", "status": "ok"}
