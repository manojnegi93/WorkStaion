import { useEffect, useRef, useState } from "react";
import { deleteDocument, listDocuments, uploadDocuments, type ChatResult, type Source } from "../api";

function matchedBy(s: Source): string {
  const parts: string[] = [];
  if (s.dense_rank) parts.push(`meaning #${s.dense_rank}`);
  if (s.sparse_rank) parts.push(`keywords #${s.sparse_rank}`);
  return parts.length ? `Matched by ${parts.join(" and ")}` : "";
}

type SourcesProps = { result: ChatResult | null; active: number | null; onSelect: (id: number | null) => void };

export function SourcesPanel({ result, active, onSelect }: SourcesProps) {
  const refs = useRef<Record<number, HTMLLIElement | null>>({});
  useEffect(() => {
    if (active) refs.current[active]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  if (!result) return <p className="muted">Sources for an answer appear here. Select an answer to see them.</p>;
  if (!result.sources.length) return <p className="muted">This answer didn't use the knowledge base.</p>;
  return (
    <>
      {result.search_query && result.search_query !== "" && (
        <p className="muted small">Searched for “{result.search_query}”</p>
      )}
      <ol className="sources">
        {result.sources.map((s) => (
          <li
            key={s.chunk_id}
            ref={(el) => (refs.current[s.id] = el)}
            className={`source${active === s.id ? " source-active" : ""}`}
          >
            <button type="button" className="source-head" onClick={() => onSelect(active === s.id ? null : s.id)}>
              <span className="source-num">{s.id}</span>
              <span>
                <span className="source-title">{s.section || s.title}</span>
                <span className="source-file">{s.source}</span>
              </span>
            </button>
            {active === s.id && (
              <div className="source-body">
                <p>{s.text}</p>
                <p className="muted small">{matchedBy(s)}</p>
              </div>
            )}
          </li>
        ))}
      </ol>
    </>
  );
}

export function KnowledgePanel({ onChange }: { onChange: () => void }) {
  const [docs, setDocs] = useState<{ source: string; chunks: number }[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const refresh = () =>
    listDocuments()
      .then(setDocs)
      .catch(() => setStatus("Couldn't load documents. Is the backend running?"));
  useEffect(() => {
    refresh();
  }, []);

  async function onFiles(files: FileList | null) {
    if (!files?.length) return;
    setStatus(`Adding ${files.length} file${files.length > 1 ? "s" : ""}…`);
    try {
      const res = await uploadDocuments(files);
      const n = Object.values(res).reduce((a, b) => a + b, 0);
      setStatus(`Added ${Object.keys(res).length} file${Object.keys(res).length > 1 ? "s" : ""} as ${n} passages.`);
      await refresh();
      onChange();
    } catch (e) {
      setStatus((e as Error).message);
    } finally {
      if (input.current) input.current.value = "";
    }
  }

  async function remove(source: string) {
    await deleteDocument(source);
    setStatus(`Removed ${source}.`);
    await refresh();
    onChange();
  }

  return (
    <>
      <ul className="docs">
        {docs.map((d) => (
          <li key={d.source}>
            <span className="doc-name">{d.source}</span>
            <span className="muted small">{d.chunks} passages</span>
            <button type="button" className="link-button" onClick={() => remove(d.source)} aria-label={`Remove ${d.source}`}>
              Remove
            </button>
          </li>
        ))}
        {!docs.length && <li className="muted">No documents yet. Add Markdown, text or PDF files.</li>}
      </ul>
      <label className="upload">
        <input ref={input} type="file" multiple accept=".md,.markdown,.txt,.pdf" onChange={(e) => onFiles(e.target.files)} />
        Add documents
      </label>
      {status && <p className="muted small" role="status">{status}</p>}
    </>
  );
}
