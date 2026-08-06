import asyncpg
from asyncpg import Pool
from pgvector.asyncpg import register_vector

_pool: Pool | None = None


async def _init_connection(conn) -> None:
    await register_vector(conn)


async def connect(database_url: str) -> None:
    global _pool
    # Install the extension before the pool so register_vector (called in init)
    # can find the public.vector type. The pool's init callback runs per-connection
    # and would fail if the extension doesn't exist yet.
    bootstrap = await asyncpg.connect(database_url)
    try:
        await bootstrap.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await bootstrap.close()

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
                -- Generated column: PostgreSQL keeps this in sync automatically on every INSERT/UPDATE.
                content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        ''')
        # HNSW: builds incrementally (no pre-existing data needed, unlike ivfflat).
        # m=16: connections per node per layer. ef_construction=64: search depth during build.
        # Higher m/ef = better recall, more RAM. Defaults are solid for learning-scale data.
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        ''')
        # GIN index for fast full-text search on the tsvector column.
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS documents_tsv_idx
            ON documents USING gin (content_tsv)
        ''')
