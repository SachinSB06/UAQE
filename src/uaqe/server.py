"""UAQE Master Backend Server.

FastAPI application providing REST endpoints, SSE streams, and static frontend hosting.
"""

from __future__ import annotations

import os
import sys
import argparse
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartException

# Ensure src is in sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

REPO_ROOT = os.path.dirname(SRC_DIR)


def init_runtime_directories() -> None:
    """Ensure essential runtime output and scratch directories exist on startup."""
    runtime_dirs = [
        os.path.join(REPO_ROOT, "output", "jobs"),
        os.path.join(REPO_ROOT, "output", "uploads", "models"),
        os.path.join(REPO_ROOT, "output", "uploads", "datasets"),
        os.path.join(REPO_ROOT, "scratch"),
    ]
    for d in runtime_dirs:
        os.makedirs(d, exist_ok=True)


# Initialize runtime directories upon module load as fail-safe
init_runtime_directories()

from uaqe.api.routes_jobs import router as jobs_router
from uaqe.api.routes_uploads import router as uploads_router

app = FastAPI(
    title="UAQE — Universal AI Quantization Engine API",
    description="Autonomous Model Optimization, Evaluation, and Telemetry API",
    version="1.0.0"
)


@app.on_event("startup")
def on_startup():
    """Ensure required runtime directories exist when server starts."""
    init_runtime_directories()


@app.exception_handler(MultiPartException)
async def multipart_exception_handler(request: Request, exc: MultiPartException):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "DATASET_UPLOAD_MULTIPART_ERROR",
                "message": "Invalid multipart request or limits exceeded.",
                "details": str(exc)
            }
        }
    )

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(jobs_router)
app.include_router(uploads_router)

# Mount built frontend if dist exists
FRONTEND_DIST = os.path.join(REPO_ROOT, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


def main():
    parser = argparse.ArgumentParser(description="UAQE Master API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    init_runtime_directories()

    print(f"============================================================")
    print(f"UAQE MASTER API SERVER — STARTING")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"============================================================")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

