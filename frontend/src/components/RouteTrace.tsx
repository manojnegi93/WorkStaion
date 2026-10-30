import type { Span } from "../api";

const LABELS: Record<string, string> = {
  input_guard: "Input check",
  rewrite_query: "Rephrase",
  retrieve: "Search",
  agent: "Think",
  tools: "Tool",
  output_guard: "Output check",
};

function detail(span: Span): string | null {
  const m = span.metadata ?? {};
  if (span.name === "retrieve") return `${m.in_context ?? 0} passages`;
  if (span.name === "tools") return (m.results ?? []).map((r: any) => r.tool.replace(/_/g, " ")).join(", ");
  if (span.name === "agent" && m.tool_calls?.length) return "calls a tool";
  if (span.name === "agent") return "writes answer";
  if (span.name === "output_guard" && typeof m.groundedness === "number")
    return `${Math.round(m.groundedness * 100)}% grounded`;
  if (span.name === "input_guard" && m.allowed === false) return "stopped";
  if (span.name === "rewrite_query" && m.rewritten) return "used history";
  return null;
}

type Props = { spans: Span[]; live?: boolean; totalMs?: number };

/** The path a question took through the LangGraph workflow, drawn as a route with stops. */
export function RouteTrace({ spans, live, totalMs }: Props) {
  if (!spans.length) return null;
  return (
    <figure className="route" aria-label="How this answer was produced">
      <ol className="route-line">
        {spans.map((s, i) => {
          const isDetour = s.name === "tools";
          const flagged = s.name === "input_guard" && s.metadata?.allowed === false;
          const d = detail(s);
          return (
            <li
              key={`${s.name}-${i}`}
              className={`stop${isDetour ? " stop-detour" : ""}${flagged ? " stop-flagged" : ""}`}
            >
              <span className="stop-dot" aria-hidden />
              <span className="stop-name">{LABELS[s.name] ?? s.name}</span>
              <span className="stop-ms">{s.duration_ms < 1 ? "<1" : Math.round(s.duration_ms)} ms</span>
              {d && <span className="stop-detail">{d}</span>}
            </li>
          );
        })}
        {live && (
          <li className="stop stop-pending">
            <span className="stop-dot" aria-hidden />
            <span className="stop-name">Working</span>
          </li>
        )}
      </ol>
      {totalMs !== undefined && !live && (
        <figcaption>Finished in {(totalMs / 1000).toFixed(totalMs < 1000 ? 2 : 1)} s</figcaption>
      )}
    </figure>
  );
}
