from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Report process liveness without checking external dependencies."""
    return {"status": "ok"}
