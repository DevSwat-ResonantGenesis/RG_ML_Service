"""
Owner-Only ML Training Module
Protected ML training functionality accessible ONLY to platform owner/creator.
Integrates ResonantGenesis_V8 training logic with owner authentication.
"""

import os
import json
import time
import hashlib
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import jwt

from .db import get_session
from .models import TrainingJob, ModelRegistry, ModelVersion


router = APIRouter(prefix="/owner/ml-training", tags=["Owner ML Training"])
security = HTTPBearer()

# Owner credentials from environment variables
OWNER_EMAIL = os.getenv("OWNER_EMAIL", "")
OWNER_JWT_SECRET = os.getenv("OWNER_JWT_SECRET", os.getenv("JWT_SECRET_KEY", "fallback-secret") + "_owner")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# External ML Database connection
ML_DB_HOST = os.getenv("ML_EXTERNAL_DB_HOST", "")
ML_DB_PORT = os.getenv("ML_EXTERNAL_DB_PORT", "5432")
ML_DB_NAME = os.getenv("ML_EXTERNAL_DB_NAME", "ml_training")
ML_DB_USER = os.getenv("ML_EXTERNAL_DB_USER", "")
ML_DB_PASSWORD = os.getenv("ML_EXTERNAL_DB_PASSWORD", "")


# ============================================
# OWNER AUTHENTICATION DEPENDENCY
# ============================================

async def require_owner(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to require owner authentication for all endpoints.
    Only platform owner/creator can access these endpoints.
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            OWNER_JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        
        # Verify this is an owner token
        if payload.get("role") != "owner" or payload.get("type") != "owner_access":
            raise HTTPException(
                status_code=403, 
                detail="Access denied. Owner privileges required."
            )
        
        return {"email": payload.get("sub", ""), "role": "owner"}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class VocabWord(BaseModel):
    word: str
    frequency: Optional[int] = 1
    category: Optional[str] = None


class TrainingDataset(BaseModel):
    name: str
    description: Optional[str] = None
    words: List[str]
    anchors: Optional[List[Dict[str, Any]]] = None


class RetrainRequest(BaseModel):
    dataset_id: Optional[str] = None
    add_words: Optional[List[str]] = None
    remove_words: Optional[List[str]] = None


class AnchorRequest(BaseModel):
    hash: str
    words: List[str]  # Must be exactly 12 words


class ForbiddenWordsRequest(BaseModel):
    add: Optional[List[str]] = None
    remove: Optional[List[str]] = None


class TrainingStatsResponse(BaseModel):
    total_vocab_size: int
    total_anchors: int
    total_datasets: int
    total_training_jobs: int
    forbidden_words_count: int
    last_training_at: Optional[str] = None


class VocabResponse(BaseModel):
    words: List[str]
    total: int


class AnchorResponse(BaseModel):
    hash: str
    words: List[str]
    xyz: List[float]
    resonance: float
    proof: Dict[str, Any]


# ============================================
# RESONANT GENESIS V8 TRAINING LOGIC
# ============================================

def clean_hex(h: str) -> str:
    """Clean and normalize hex string."""
    h = h.lower().strip()
    if h.startswith('0x'):
        h = h[2:]
    return ''.join(c for c in h if c in '0123456789abcdef')


def hash_xyz(h: str) -> List[float]:
    """Convert hash to 3D coordinates."""
    n = int(h, 16)
    mask53 = (1 << 53) - 1
    x = (n >> 107) & mask53
    y = (n >> 54) & mask53
    z = n & ((1 << 54) - 1)
    
    def norm(v, m):
        return (2 * v / m) - 1.0
    
    return [float(norm(x, mask53)), float(norm(y, mask53)), float(norm(z, (1 << 54) - 1))]


def resonance(xyz: List[float]) -> float:
    """Calculate resonance value from coordinates."""
    import math
    x, y, z = xyz
    return math.sin(0.7 * x) + math.cos(0.91 * y) + math.tan(0.81 * z)


def merkle_root(words: List[str]) -> str:
    """Calculate merkle root of word list."""
    leaves = [hashlib.sha256(w.encode()).hexdigest() for w in words]
    if not leaves:
        return ''
    layer = leaves
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer), 2):
            a = layer[i]
            b = layer[i + 1] if i + 1 < len(layer) else layer[i]
            nxt.append(hashlib.sha256((a + b).encode()).hexdigest())
        layer = nxt
    return layer[0]


def gen_words(seed_hex: str, vocab: List[str], forbidden: List[str]) -> List[str]:
    """Generate 12 words from seed hash."""
    rnd = random.Random(int(seed_hex, 16))
    pool = [w for w in vocab if w not in forbidden]
    rnd.shuffle(pool)
    out = []
    for w in pool:
        if len(out) == 12:
            break
        if w not in out:
            out.append(w)
    while len(out) < 12:
        out.append(f'token{rnd.randint(0, 9999)}')
    return out


# ============================================
# IN-MEMORY STORAGE (Will be replaced with DB)
# ============================================

# These will be migrated to external ML database
_vocab_store: List[str] = [
    "guide", "wolf", "artist", "signal", "ring", "orbit", "align", "vector",
    "resonance", "field", "token", "anchor", "state", "clean", "prepare",
    "today", "future", "genesis", "quantum", "neural", "semantic", "identity"
]
_anchors_store: List[Dict[str, Any]] = []
_forbidden_store: List[str] = ["entropy", "phase", "cluster"]
_datasets_store: List[Dict[str, Any]] = []


# ============================================
# OWNER-ONLY ENDPOINTS
# ============================================

@router.get("/status")
async def get_training_status(owner: dict = Depends(require_owner)):
    """Get ML training system status. Owner only."""
    return {
        "status": "operational",
        "owner": owner["email"],
        "vocab_size": len(_vocab_store),
        "anchors_count": len(_anchors_store),
        "forbidden_count": len(_forbidden_store),
        "external_db_configured": bool(ML_DB_HOST),
    }


@router.get("/stats", response_model=TrainingStatsResponse)
async def get_training_stats(
    owner: dict = Depends(require_owner),
    session: AsyncSession = Depends(get_session)
):
    """Get comprehensive training statistics. Owner only."""
    # Get training job count from database
    result = await session.execute(select(func.count(TrainingJob.id)))
    training_jobs_count = result.scalar() or 0
    
    # Get last training timestamp
    result = await session.execute(
        select(TrainingJob.completed_at)
        .where(TrainingJob.status == "completed")
        .order_by(TrainingJob.completed_at.desc())
        .limit(1)
    )
    last_training = result.scalar()
    
    return TrainingStatsResponse(
        total_vocab_size=len(_vocab_store),
        total_anchors=len(_anchors_store),
        total_datasets=len(_datasets_store),
        total_training_jobs=training_jobs_count,
        forbidden_words_count=len(_forbidden_store),
        last_training_at=last_training.isoformat() if last_training else None
    )


@router.get("/vocab", response_model=VocabResponse)
async def get_vocabulary(owner: dict = Depends(require_owner)):
    """Get current vocabulary. Owner only."""
    return VocabResponse(words=sorted(_vocab_store), total=len(_vocab_store))


@router.post("/vocab")
async def add_vocabulary(
    words: List[str],
    owner: dict = Depends(require_owner)
):
    """Add words to vocabulary. Owner only."""
    added = []
    for word in words:
        word = word.lower().strip()
        if word and word not in _vocab_store and word not in _forbidden_store:
            _vocab_store.append(word)
            added.append(word)
    
    return {
        "added": added,
        "total_vocab_size": len(_vocab_store)
    }


@router.delete("/vocab")
async def remove_vocabulary(
    words: List[str],
    owner: dict = Depends(require_owner)
):
    """Remove words from vocabulary. Owner only."""
    removed = []
    for word in words:
        word = word.lower().strip()
        if word in _vocab_store:
            _vocab_store.remove(word)
            removed.append(word)
    
    return {
        "removed": removed,
        "total_vocab_size": len(_vocab_store)
    }


@router.get("/forbidden")
async def get_forbidden_words(owner: dict = Depends(require_owner)):
    """Get forbidden words list. Owner only."""
    return {"forbidden": sorted(_forbidden_store), "total": len(_forbidden_store)}


@router.post("/forbidden")
async def update_forbidden_words(
    request: ForbiddenWordsRequest,
    owner: dict = Depends(require_owner)
):
    """Update forbidden words list. Owner only."""
    added = []
    removed = []
    
    if request.add:
        for word in request.add:
            word = word.lower().strip()
            if word and word not in _forbidden_store:
                _forbidden_store.append(word)
                added.append(word)
                # Also remove from vocab if present
                if word in _vocab_store:
                    _vocab_store.remove(word)
    
    if request.remove:
        for word in request.remove:
            word = word.lower().strip()
            if word in _forbidden_store:
                _forbidden_store.remove(word)
                removed.append(word)
    
    return {
        "added": added,
        "removed": removed,
        "forbidden": sorted(_forbidden_store)
    }


@router.get("/anchors")
async def get_anchors(owner: dict = Depends(require_owner)):
    """Get all anchors. Owner only."""
    return {"anchors": _anchors_store, "total": len(_anchors_store)}


@router.post("/anchors", response_model=AnchorResponse)
async def add_anchor(
    request: AnchorRequest,
    owner: dict = Depends(require_owner)
):
    """Add a new anchor. Owner only."""
    key = clean_hex(request.hash)
    
    if len(request.words) != 12:
        raise HTTPException(status_code=400, detail="Anchor must have exactly 12 words")
    
    # Check for duplicate
    if any(clean_hex(a['hash']) == key for a in _anchors_store):
        raise HTTPException(status_code=409, detail="Anchor already exists")
    
    xyz = hash_xyz(key)
    r = resonance(xyz)
    
    entry = {
        'hash': '0x' + key,
        'words': request.words,
        'xyz': xyz,
        'resonance': r,
        'proof': {
            'root': merkle_root(request.words),
            'ts': int(time.time())
        }
    }
    
    _anchors_store.append(entry)
    
    return AnchorResponse(
        hash=entry['hash'],
        words=entry['words'],
        xyz=entry['xyz'],
        resonance=entry['resonance'],
        proof=entry['proof']
    )


@router.delete("/anchors/{anchor_hash}")
async def delete_anchor(
    anchor_hash: str,
    owner: dict = Depends(require_owner)
):
    """Delete an anchor. Owner only."""
    key = clean_hex(anchor_hash)
    
    for i, anchor in enumerate(_anchors_store):
        if clean_hex(anchor['hash']) == key:
            _anchors_store.pop(i)
            return {"deleted": True, "hash": anchor_hash}
    
    raise HTTPException(status_code=404, detail="Anchor not found")


@router.post("/retrain")
async def retrain_model(
    request: RetrainRequest,
    owner: dict = Depends(require_owner),
    session: AsyncSession = Depends(get_session)
):
    """Retrain the model with current vocabulary. Owner only."""
    # Add new words if provided
    if request.add_words:
        for word in request.add_words:
            word = word.lower().strip()
            if word and word not in _vocab_store and word not in _forbidden_store:
                _vocab_store.append(word)
    
    # Remove words if provided
    if request.remove_words:
        for word in request.remove_words:
            word = word.lower().strip()
            if word in _vocab_store:
                _vocab_store.remove(word)
    
    # Create training job record
    job = TrainingJob(
        name=f"Owner Retrain {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        status="completed",
        config={"vocab_size": len(_vocab_store), "triggered_by": owner["email"]},
        metrics={"vocab_size": len(_vocab_store), "forbidden_count": len(_forbidden_store)},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    return {
        "message": "Model retrained successfully",
        "job_id": str(job.id),
        "vocab_size": len(_vocab_store),
        "forbidden_count": len(_forbidden_store)
    }


@router.post("/predict")
async def predict_words(
    hash_value: str,
    owner: dict = Depends(require_owner)
):
    """Predict words for a hash. Owner only."""
    key = clean_hex(hash_value)
    
    if not key:
        raise HTTPException(status_code=400, detail="Invalid hash")
    
    # Check if anchored
    for anchor in _anchors_store:
        if clean_hex(anchor['hash']) == key:
            return {
                'hash': '0x' + key,
                'words': anchor['words'],
                'xyz': anchor.get('xyz', hash_xyz(key)),
                'resonance': anchor.get('resonance', 0.0),
                'anchor': True
            }
    
    # Generate prediction
    xyz = hash_xyz(key)
    r = resonance(xyz)
    words = gen_words(key, _vocab_store, _forbidden_store)
    
    return {
        'hash': '0x' + key,
        'words': words,
        'xyz': xyz,
        'resonance': r,
        'anchor': False
    }


@router.get("/datasets")
async def get_datasets(owner: dict = Depends(require_owner)):
    """Get all training datasets. Owner only."""
    return {"datasets": _datasets_store, "total": len(_datasets_store)}


@router.post("/datasets")
async def create_dataset(
    dataset: TrainingDataset,
    owner: dict = Depends(require_owner)
):
    """Create a new training dataset. Owner only."""
    import uuid
    
    entry = {
        "id": str(uuid.uuid4()),
        "name": dataset.name,
        "description": dataset.description,
        "words": dataset.words,
        "anchors": dataset.anchors or [],
        "created_at": datetime.utcnow().isoformat(),
        "created_by": owner["email"]
    }
    
    _datasets_store.append(entry)
    
    return entry


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    owner: dict = Depends(require_owner)
):
    """Delete a training dataset. Owner only."""
    for i, dataset in enumerate(_datasets_store):
        if dataset['id'] == dataset_id:
            _datasets_store.pop(i)
            return {"deleted": True, "id": dataset_id}
    
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.get("/health")
async def health():
    """Health check - no auth required."""
    return {"service": "owner-ml-training", "status": "ok", "protected": True}
