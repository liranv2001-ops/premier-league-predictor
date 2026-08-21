import { useState } from "react";

interface Props {
  assumptions: string[];
}

/**
 * The caveats the model shipped with its numbers.
 *
 * These live in the payload as a machine-readable array precisely so the UI can show
 * them. A 63% title probability and a shortlist of named players look authoritative;
 * "squads carried forward, ~47% of players change club each summer" is the context that
 * makes them honest. Collapsible, but open by default - a caveat behind a click is a
 * caveat nobody reads.
 */
export default function AssumptionsBanner({ assumptions }: Props) {
  const [open, setOpen] = useState(true);
  if (assumptions.length === 0) return null;

  return (
    <aside
      className="rounded-xl px-4 py-3"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center gap-2 text-left text-sm font-medium"
        style={{ color: "var(--text-primary)" }}
      >
        <span aria-hidden="true">ⓘ</span>
        How to read these predictions
        <span className="ml-auto text-xs" style={{ color: "var(--text-muted)" }}>
          {open ? "Hide" : `${assumptions.length} notes`}
        </span>
      </button>

      {open && (
        <ul className="mt-2.5 space-y-1.5 pl-6 text-sm" style={{ color: "var(--text-secondary)" }}>
          {assumptions.map((note) => (
            <li key={note} className="list-disc">
              {note}
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
