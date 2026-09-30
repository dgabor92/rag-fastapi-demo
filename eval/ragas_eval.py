#!/usr/bin/env python3
"""
RAG Evaluation — Phase 5 (RAGAS variant)
Uses the RAGAS framework with Claude Haiku as the LLM judge.

Usage:
  pip install -r eval/requirements-eval.txt
  python eval/ragas_eval.py [--api-url http://localhost:8000]

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
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper

# answer_relevancy is intentionally excluded: it requires sentence-transformers
# which has a scipy binary incompatibility on macOS Darwin 27+.
# To add it back: fix scipy with `pip install --no-binary :all: scipy`, then:
#   from langchain_community.embeddings import HuggingFaceEmbeddings
#   from ragas.embeddings import LangchainEmbeddingsWrapper
#   from ragas.metrics import answer_relevancy

load_dotenv(Path(__file__).parent.parent / '.env')

JUDGE_MODEL = 'claude-haiku-4-5-20251001'


def build_ragas_llm(api_key: str):
    langchain_llm = ChatAnthropic(model=JUDGE_MODEL, api_key=api_key)
    return LangchainLLMWrapper(langchain_llm)


def query_rag(http: httpx.Client, api_base: str, question: str) -> dict:
    resp = http.post(
        f'{api_base}/query',
        json={'question': question, 'top_k': 3},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def ingest_corpus(http: httpx.Client, api_base: str, corpus_path: Path) -> int:
    text = corpus_path.read_text()
    resp = http.post(f'{api_base}/ingest', json={'text': text}, timeout=60)
    resp.raise_for_status()
    return resp.json()['chunks_stored']


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG Evaluation with RAGAS')
    parser.add_argument('--api-url', default='http://localhost:8000', help='FastAPI base URL')
    args = parser.parse_args()

    api_base = args.api_url.rstrip('/')
    corpus_path = Path(__file__).parent / 'test_corpus.txt'
    cases_path = Path(__file__).parent / 'test_cases.json'

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set in .env')
        sys.exit(1)

    ragas_llm = build_ragas_llm(api_key)

    faithfulness.llm = ragas_llm
    context_precision.llm = ragas_llm

    test_cases = json.loads(cases_path.read_text())

    with httpx.Client() as http:
        try:
            http.get(f'{api_base}/health', timeout=5).raise_for_status()
            print(f'API reachable at {api_base}')
        except Exception:
            print(f'ERROR: API not reachable at {api_base}')
            sys.exit(1)

        print('Ingesting test corpus...')
        chunks = ingest_corpus(http, api_base, corpus_path)
        print(f'Ingested {chunks} chunks\n')

        questions, answers, contexts, ground_truths = [], [], [], []

        for case in test_cases:
            print(f'Querying Q{case["id"]}: {case["question"][:60]}...' if len(case['question']) > 60 else f'Querying Q{case["id"]}: {case["question"]}')
            result = query_rag(http, api_base, case['question'])
            questions.append(case['question'])
            answers.append(result['answer'])
            contexts.append([s['content'] for s in result['sources']])
            ground_truths.append(case.get('ground_truth', ''))

    dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths,
    })

    print('\nRunning RAGAS evaluation...')
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision],
    )

    print('\n' + '=' * 70)
    print('  RAGAS EVALUATION REPORT')
    print('=' * 70)
    print(result)

    results_path = Path(__file__).parent / 'ragas_results.json'
    result.to_pandas().to_json(results_path, orient='records', indent=2, force_ascii=False)
    print(f'\nFull results saved -> {results_path}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
