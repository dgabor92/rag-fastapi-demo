from fastapi import APIRouter, HTTPException
from app.models import HybridQueryRequest, HybridQueryResponse, HybridChunkMatch
from app.embed import embed
from app.db import get_pool
from app.llm import client

router = APIRouter()


def _rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


@router.post('/hybrid-query', response_model=HybridQueryResponse)
async def hybrid_query(body: HybridQueryRequest) -> HybridQueryResponse:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail='question cannot be empty')

    vector = await embed(body.question)
    pool = get_pool()
    candidate_k = body.top_k * 2

    async with pool.acquire() as conn:
        # Step 1: semantic search -- finds docs by meaning
        semantic_rows = await conn.fetch(
            '''
            SELECT id, content
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT $2
            ''',
            vector,
            candidate_k,
        )

        # Step 2: full-text search -- finds docs by exact keywords
        fulltext_rows = await conn.fetch(
            '''
            SELECT id, content
            FROM documents
            WHERE content_tsv @@ websearch_to_tsquery('english', $1)
            ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', $1)) DESC
            LIMIT $2
            ''',
            body.question,
            candidate_k,
        )

    # Step 3: RRF -- merge the two ranked lists into one score
    # Each doc gets 1/(k + rank) from each list it appears in.
    # A doc appearing in both lists scores higher than one appearing in only one.
    scores: dict[int, float] = {}
    contents: dict[int, str] = {}

    for rank, row in enumerate(semantic_rows, start=1):
        scores[row['id']] = scores.get(row['id'], 0) + _rrf_score(rank, body.rrf_k)
        contents[row['id']] = row['content']

    for rank, row in enumerate(fulltext_rows, start=1):
        scores[row['id']] = scores.get(row['id'], 0) + _rrf_score(rank, body.rrf_k)
        contents[row['id']] = row['content']

    # Step 4: sort by combined score, take top_k
    top_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[: body.top_k]

    if not top_ids:
        raise HTTPException(status_code=404, detail='No documents ingested yet')

    sources = [
        HybridChunkMatch(content=contents[doc_id], rrf_score=round(scores[doc_id], 6))
        for doc_id in top_ids
    ]

    context = '\n\n'.join(s.content for s in sources)

    message = await client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=1024,
        messages=[
            {
                'role': 'user',
                'content': f'Context:\n{context}\n\nQuestion: {body.question}\n\nAnswer based only on the context above.',
            }
        ],
    )

    return HybridQueryResponse(answer=message.content[0].text, sources=sources)
