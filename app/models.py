from pydantic import BaseModel


class IngestRequest(BaseModel):
    text: str


class IngestResponse(BaseModel):
    chunks_stored: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class ChunkMatch(BaseModel):
    content: str
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkMatch]


class HybridQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    # RRF constant: dampens the effect of top ranks. 60 is the standard default.
    rrf_k: int = 60


class HybridChunkMatch(BaseModel):
    content: str
    rrf_score: float


class HybridQueryResponse(BaseModel):
    answer: str
    sources: list[HybridChunkMatch]


class HealthResponse(BaseModel):
    status: str
    db: str
