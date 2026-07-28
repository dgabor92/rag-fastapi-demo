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


class HealthResponse(BaseModel):
    status: str
    db: str
