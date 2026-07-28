import asyncpg
from asyncpg import Pool
from pgvector.asyncpg import register_vector

_pool: Pool | None = None


async def _init_connection(conn) -> None:
    await register_vector(conn)


async def connect(database_url: str) -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )
    await _init_schema()


async def disconnect() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> Pool:
    if _pool is None:
        raise RuntimeError('DB pool not initialised')
    return _pool


async def _init_schema() -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(768) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        ''')
