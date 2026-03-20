# RG ML Service

> **Part of the [ResonantGenesis](https://dev-swat.com) platform** — Machine learning model training and inference service.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![Port: 8000](https://img.shields.io/badge/Port-8000-orange.svg)]()
[![Database: PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

ML service for model management, owner-directed training, and inference pipelines. Supports custom model training and deployment for platform-specific use cases.

## Features

- **Model management** — Register, version, and serve ML models
- **Owner training** — Platform owner can trigger custom model training runs
- **Inference** — Real-time model inference endpoints

## Quick Start

```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/ml"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Deployment Status

- **Extracted from**: `genesis2026_production_backend/ml_service/`
- **Server path**: `/home/deploy/RG_ML_Service`
- **Docker service**: `ml_service`

---
**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [dev-swat.com](https://dev-swat.com)
