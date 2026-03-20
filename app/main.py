import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import router as ml_router
from .owner_training import router as owner_training_router

# Deterministic sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Single service entrypoint
app = FastAPI(
    title="Ml_Service Service",
    description="Service for Genesis2026",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml_service"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": f"Ml_Service Service is running"}

# Service-specific endpoint
@app.get("/api/v1/status")
async def status():
    return {"service": "ml_service", "status": "active", "version": "1.0.0"}

# Include routers
app.include_router(ml_router, prefix="/api/v1")
app.include_router(owner_training_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
