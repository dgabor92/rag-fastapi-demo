#!/usr/bin/env python3
"""
RAG Evaluation Pipeline — Phase 5
LLM-as-a-judge: Faithfulness, Answer Relevancy, Context Precision.

Usage:
  python eval/eval.py [--api-url http://localhost:8000]

Prerequisites:
  docker compose up -d
  ollama serve  (in a separate terminal)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

JUDGE_MODEL = 'claude-haiku-4-5-20251001'

FAITHFULNESS_PROMPT = """\
You are an AI evaluator. Rate the FAITHFULNESS of an answer.

Question: {question}
Retrieved Context:
{context}
Answer: {answer}

FAITHFULNESS: Does every claim in the answer appear in the retrieved context?
- 1.0  every claim is directly supported by the context
- 0.5  some claims are supported, others are not in the context
- 0.0  the answer makes claims absent from the context (hallucination)

Respond ONLY with valid JSON on one line: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""

RELEVANCY_PROMPT = """\
You are an AI evaluator. Rate the ANSWER RELEVANCY.

Question: {question}
Answer: {answer}

ANSWER RELEVANCY: Does the answer directly address the question?
- 1.0  fully and directly answers the question
- 0.5  partially answers the question
- 0.0  does not answer the question

Respond ONLY with valid JSON on one line: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""

CONTEXT_PRECISION_PROMPT = """\
You are an AI evaluator. Rate the CONTEXT PRECISION.

Question: {question}
Retrieved Contexts:
{context}

CONTEXT PRECISION: Are the retrieved chunks useful for answering this question?
- 1.0  all chunks are relevant to the question
- 0.5  some chunks are relevant, some are noise
- 0.0  none of the chunks help answer the question

Respond ONLY with valid JSON on one line: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""


def judge(anthropic_client: anthropic.Anthropic, prompt: str) -> dict:
    msg = anthropic_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{'role': 'user', 'content': prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # extract first JSON object if model wrapped in prose
        start = raw.find('{')
        end = raw.rfind('}') + 1
        return json.loads(raw[start:end])


def ingest_corpus(http: httpx.Client, api_base: str, corpus_path: Path) -> int:
    text = corpus_path.read_text()
    resp = http.post(f'{api_base}/ingest', json={'text': text}, timeout=60)
    resp.raise_for_status()
    return resp.json()['chunks_stored']


def query_rag(http: httpx.Client, api_base: str, question: str) -> dict:
    resp = http.post(
        f'{api_base}/query',
        json={'question': question, 'top_k': 3},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def eval_case(
    http: httpx.Client,
    api_base: str,
    anthropic_client: anthropic.Anthropic,
    case: dict,
) -> dict:
    question = case['question']
    rag_result = query_rag(http, api_base, question)
    answer = rag_result['answer']
    context = '\n\n---\n\n'.join(s['content'] for s in rag_result['sources'])

    metrics = [
        ('faithfulness',      FAITHFULNESS_PROMPT,     dict(question=question, context=context, answer=answer)),
        ('answer_relevancy',  RELEVANCY_PROMPT,         dict(question=question, answer=answer)),
        ('context_precision', CONTEXT_PRECISION_PROMPT, dict(question=question, context=context)),
    ]
    scores = {key: judge(anthropic_client, tpl.format(**kwargs)) for key, tpl, kwargs in metrics}

    return {
        'id': case['id'],
        'question': question,
        'answer': answer,
        'sources_count': len(rag_result['sources']),
        **scores,
    }


def print_report(results: list[dict]) -> None:
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision']

    print('\n' + '=' * 70)
    print('  RAG EVALUATION REPORT')
    print('=' * 70)

    for r in results:
        print(f'\n[Q{r["id"]}] {r["question"]}')
        preview = r['answer'][:130] + '...' if len(r['answer']) > 130 else r['answer']
        print(f'  Answer  : {preview}')
        print(f'  Sources : {r["sources_count"]} chunks retrieved')
        for m in metrics:
            score = r[m]['score']
            reason = r[m]['reason']
            filled = int(round(score * 10))
            bar = '█' * filled + '░' * (10 - filled)
            label = m.replace('_', ' ').title()
            print(f'  {label:<22} [{bar}] {score:.2f}  {reason}')

    print('\n' + '-' * 70)
    print('  AVERAGES')
    for m in metrics:
        avg = sum(r[m]['score'] for r in results) / len(results)
        filled = int(round(avg * 10))
        bar = '█' * filled + '░' * (10 - filled)
        label = m.replace('_', ' ').title()
        print(f'  {label:<22} [{bar}] {avg:.2f}')
    print('=' * 70 + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG Evaluation Pipeline')
    parser.add_argument('--api-url', default='http://localhost:8000', help='FastAPI base URL')
    args = parser.parse_args()

    api_base = args.api_url.rstrip('/')
    corpus_path = Path(__file__).parent / 'test_corpus.txt'
    cases_path = Path(__file__).parent / 'test_cases.json'

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set in .env')
        sys.exit(1)

    anthropic_client = anthropic.Anthropic(api_key=api_key)
    test_cases = json.loads(cases_path.read_text())

    with httpx.Client() as http:
        try:
            http.get(f'{api_base}/health', timeout=5).raise_for_status()
            print(f'API reachable at {api_base}')
        except Exception:
            print(f'ERROR: API not reachable at {api_base}')
            print('Run: docker compose up -d  (and ollama serve in a separate terminal)')
            sys.exit(1)

        print('Ingesting test corpus...')
        chunks = ingest_corpus(http, api_base, corpus_path)
        print(f'Ingested {chunks} chunks\n')

        results = []
        for case in test_cases:
            print(f'Q{case["id"]}: {case["question"][:60]}...' if len(case['question']) > 60 else f'Q{case["id"]}: {case["question"]}')
            result = eval_case(http, api_base, anthropic_client, case)
            results.append(result)
            f = result['faithfulness']['score']
            r = result['answer_relevancy']['score']
            p = result['context_precision']['score']
            print(f'     faithfulness={f:.2f}  relevancy={r:.2f}  precision={p:.2f}')

    results_path = Path(__file__).parent / 'results.json'
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'\nFull results saved -> {results_path}')

    print_report(results)


if __name__ == '__main__':
    main()
