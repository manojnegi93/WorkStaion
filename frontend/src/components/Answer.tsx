import { Fragment, type ReactNode } from "react";

type Props = { text: string; onCite: (id: number) => void; activeCite: number | null };

const FLAG_TEXT: Record<string, string> = {
  prompt_injection: "Blocked an attempt to override the assistant's instructions.",
  unsafe_content: "Blocked an unsafe request.",
  input_too_long: "Message was too long to process.",
  low_groundedness: "Parts of this answer are weakly supported by the sources.",
  ungrounded_answer_blocked: "The draft answer wasn't supported by the sources, so it was withheld.",
  invalid_citation_removed: "Removed a citation that pointed to a missing source.",
  unsupported_scent_claim: "The draft answer described a scent the catalogue doesn't list, so it was withheld.",
  system_prompt_leak: "Withheld an answer that exposed internal instructions.",
};

export function flagText(flag: string): string {
  if (flag.startsWith("pii_redacted:")) {
    return `Hid personal data (${flag.split(":")[1].replace("_", " ")}) before it reached the model.`;
  }
  return FLAG_TEXT[flag] ?? flag;
}

function inline(text: string, onCite: Props["onCite"], active: number | null): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\[(\d+)\]|_(.+?)_|\*\*(.+?)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) {
      const id = Number(m[1]);
      out.push(
        <button
          key={m.index}
          type="button"
          className={`cite${active === id ? " cite-active" : ""}`}
          onClick={() => onCite(id)}
          aria-label={`Show source ${id}`}
        >
          {id}
        </button>,
      );
    } else if (m[2]) out.push(<em key={m.index}>{m[2]}</em>);
    else if (m[3]) out.push(<strong key={m.index}>{m[3]}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** Minimal markdown: paragraphs, "-" bullet lists, _em_, **strong**, and [n] citations. */
export function Answer({ text, onCite, activeCite }: Props) {
  const blocks = text.trim().split(/\n{2,}/);
  return (
    <div className="answer-text">
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        if (lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l))) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^\s*([-*•]|\d+\.)\s+/, ""), onCite, activeCite)}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i}>
            {lines.map((l, j) => (
              <Fragment key={j}>
                {j > 0 && <br />}
                {inline(l, onCite, activeCite)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
