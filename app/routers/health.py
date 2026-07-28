from fastapi import APIRouter
from app.models import HealthResponse
from app.db import get_pool

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        db_status = 'ok'
    except Exception:
        db_status = 'error'

    return HealthResponse(status='ok', db=db_status)
