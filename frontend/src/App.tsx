import { useState } from 'react'

const API = import.meta.env.VITE_API_URL ?? '/api'

type Source = { content: string; similarity: number }
type HybridSource = { content: string; rrf_score: number }

type QueryResult = {
  answer: string
  sources: Source[]
}

type HybridResult = {
  answer: string
  sources: HybridSource[]
}

type Tab = 'query' | 'hybrid' | 'ingest'

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono font-semibold ${color}`}>
      {children}
    </span>
  )
}

function SourceCard({ content, score, label }: { content: string; score: number; label: string }) {
  const pct = Math.round(score * 100)
  const barColor = pct > 65 ? 'bg-emerald-500' : pct > 45 ? 'bg-amber-500' : 'bg-slate-500'

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <Badge color="bg-slate-700 text-slate-300">{label}: {(score).toFixed(4)}</Badge>
        <div className="flex-1 h-1.5 rounded-full bg-slate-700 overflow-hidden">
          <div className={`h-full rounded-full ${barColor}`} style={{ width: `${Math.min(pct, 100)}%` }} />
        </div>
      </div>
      <p className="text-sm text-slate-300 leading-relaxed line-clamp-4">{content}</p>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>('query')

  const [question, setQuestion] = useState('')
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [hybridResult, setHybridResult] = useState<HybridResult | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)

  const [ingestText, setIngestText] = useState('')
  const [ingestSource, setIngestSource] = useState('')
  const [ingestLoading, setIngestLoading] = useState(false)
  const [ingestMsg, setIngestMsg] = useState<{ ok: boolean; text: string } | null>(null)

  const [error, setError] = useState<string | null>(null)

  async function handleQuery(e: React.FormEvent) {
    e.preventDefault()
    if (!question.trim()) return
    setQueryLoading(true)
    setError(null)
    setQueryResult(null)
    setHybridResult(null)
    try {
      if (tab === 'query') {
        const res = await fetch(`${API}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        })
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        setQueryResult(await res.json())
      } else {
        const res = await fetch(`${API}/hybrid-query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        })
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        setHybridResult(await res.json())
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setQueryLoading(false)
    }
  }

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault()
    if (!ingestText.trim()) return
    setIngestLoading(true)
    setIngestMsg(null)
    setError(null)
    try {
      const res = await fetch(`${API}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: ingestText, source: ingestSource || undefined }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      setIngestMsg({ ok: true, text: `${data.chunks_stored} chunk tárolva.` })
      setIngestText('')
      setIngestSource('')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIngestLoading(false)
    }
  }

  const activeResult = tab === 'query' ? queryResult : hybridResult

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">R</div>
        <div>
          <h1 className="text-lg font-semibold leading-none">RAG FastAPI Demo</h1>
          <p className="text-xs text-slate-500 mt-0.5">pgvector + Ollama + Claude Haiku</p>
        </div>
      </header>

      <div className="flex-1 max-w-3xl mx-auto w-full px-4 py-8 space-y-6">
        <div className="flex gap-1 bg-slate-900 rounded-xl p-1 border border-slate-800">
          {(['query', 'hybrid', 'ingest'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => { setTab(t); setError(null) }}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                tab === t
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {t === 'query' && 'Semantic Query'}
              {t === 'hybrid' && 'Hybrid Query'}
              {t === 'ingest' && 'Ingest'}
            </button>
          ))}
        </div>

        {(tab === 'query' || tab === 'hybrid') && (
          <form onSubmit={handleQuery} className="space-y-3">
            <div className="relative">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Tedd fel a kérdésed..."
                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 pr-28 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <button
                type="submit"
                disabled={queryLoading || !question.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-1.5 rounded-lg transition-colors cursor-pointer"
              >
                {queryLoading ? 'Keresés...' : 'Keresés'}
              </button>
            </div>
            {tab === 'hybrid' && (
              <p className="text-xs text-slate-500">Vektoros + full-text keresés, RRF fusion-nal kombinálva.</p>
            )}
          </form>
        )}

        {tab === 'ingest' && (
          <form onSubmit={handleIngest} className="space-y-3">
            <textarea
              value={ingestText}
              onChange={(e) => setIngestText(e.target.value)}
              placeholder="Illeszd be a szöveget amit be szeretnél tölteni a tudásbázisba..."
              rows={8}
              className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
            />
            <input
              value={ingestSource}
              onChange={(e) => setIngestSource(e.target.value)}
              placeholder="Forrás (opcionális, pl. docs/readme.md)"
              className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              type="submit"
              disabled={ingestLoading || !ingestText.trim()}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-sm font-semibold py-3 rounded-xl transition-colors cursor-pointer"
            >
              {ingestLoading ? 'Töltés...' : 'Betöltés a tudásbázisba'}
            </button>
            {ingestMsg && (
              <p className={`text-sm text-center ${ingestMsg.ok ? 'text-emerald-400' : 'text-red-400'}`}>
                {ingestMsg.text}
              </p>
            )}
          </form>
        )}

        {error && (
          <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {activeResult && (
          <div className="space-y-4">
            <div className="rounded-xl border border-slate-700 bg-slate-900 p-5">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-2 font-semibold">Válasz</p>
              <p className="text-slate-100 leading-relaxed whitespace-pre-wrap">{activeResult.answer}</p>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                Forrás chunkok ({activeResult.sources.length})
              </p>
              {tab === 'query' &&
                (activeResult as QueryResult).sources.map((s, i) => (
                  <SourceCard key={i} content={s.content} score={s.similarity} label="similarity" />
                ))}
              {tab === 'hybrid' &&
                (activeResult as HybridResult).sources.map((s, i) => (
                  <SourceCard key={i} content={s.content} score={s.rrf_score} label="rrf" />
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
