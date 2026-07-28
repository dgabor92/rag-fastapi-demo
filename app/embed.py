import httpx
from app.config import settings


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f'{settings.ollama_url}/api/embeddings',
            json={'model': settings.embed_model, 'prompt': text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()['embedding']


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks
