export type Source = {
  id: number;
  chunk_id: string;
  source: string;
  section: string;
  title: string;
  text: string;
  score: number;
  dense_rank: number | null;
  sparse_rank: number | null;
};

export type Span = {
  name: string;
  start_ms: number;
  duration_ms: number;
  metadata: Record<string, any>;
};

export type ChatResult = {
  answer: string;
  sources: Source[];
  flags: string[];
  blocked: boolean;
  groundedness: number | null;
  search_query: string | null;
  model: string;
  prompt_version: string;
  trace: {
    trace_id: string;
    total_ms: number;
    llm_calls: number;
    tokens: { input: number; output: number; total: number };
    spans: Span[];
  };
};

export type Health = {
  status: string;
  llm_provider: string;
  chat_model: string;
  embedding_model: string;
  vector_store: string;
  chunks_indexed: number;
  tools: string[];
};

export type Turn = { role: "user" | "assistant"; content: string };

type StreamHandlers = {
  onStep: (node: string, span: Span | null) => void;
  onToken: (text: string) => void;
  onFinal: (result: ChatResult) => void;
  onError: (message: string) => void;
};

/** POST /api/chat/stream and parse Server-Sent Events from the response body. */
export async function streamChat(question: string, history: Turn[], h: StreamHandlers, signal?: AbortSignal) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok || !res.body) {
    h.onError(`The assistant service returned ${res.status}. Check that the backend is running on port 8000.`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const event = /^event: (.*)$/m.exec(raw)?.[1];
      const data = /^data: (.*)$/m.exec(raw)?.[1];
      if (!event || !data) continue;
      const payload = JSON.parse(data);
      if (event === "step") h.onStep(payload.node, payload.span?.name ? payload.span : null);
      else if (event === "token") h.onToken(payload.text);
      else if (event === "final") h.onFinal(payload);
      else if (event === "error") h.onError(payload.message);
    }
  }
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

export async function listDocuments(): Promise<{ source: string; chunks: number }[]> {
  const r = await fetch("/api/documents");
  return (await r.json()).documents;
}

export async function uploadDocuments(files: FileList): Promise<Record<string, number>> {
  const form = new FormData();
  Array.from(files).forEach((f) => form.append("files", f));
  const r = await fetch("/api/documents", { method: "POST", body: form });
  const body = await r.json();
  if (!r.ok) throw new Error(body.detail ?? `Upload failed with ${r.status}`);
  return body.ingested;
}

export async function deleteDocument(source: string) {
  await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
}
