import { useState } from "react";
import type { LogoMapping, PlayerMapping } from "../lib/assets";

interface Props {
  players: PlayerMapping | null;
  logos: LogoMapping | null;
}

/**
 * Photo and badge credits.
 *
 * This is a licence obligation, not a nicety. Every player photo here is CC BY-SA or
 * CC BY, both of which require attributing the photographer and naming the licence.
 * Recording that in `mapping.json` satisfies traceability but not attribution — the
 * credit has to appear wherever the image does, which is here.
 */
export default function Credits({ players, logos }: Props) {
  const [open, setOpen] = useState(false);
  const credits = players ? Object.entries(players.players) : [];
  const placeholders = players?.placeholders ?? [];

  if (credits.length === 0 && !logos) return null;

  return (
    <div className="mt-4">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="cursor-pointer text-xs underline underline-offset-2"
        style={{ color: "var(--text-secondary)" }}
      >
        {open ? "Hide" : "Show"} image credits and licences ({credits.length} photos)
      </button>

      {open && (
        <div className="mt-3 space-y-3 text-xs" style={{ color: "var(--text-muted)" }}>
          {logos && <p>{logos.note}</p>}

          {credits.length > 0 && (
            <div>
              <p className="mb-1.5" style={{ color: "var(--text-secondary)" }}>
                Player photographs from Wikimedia Commons:
              </p>
              <ul className="space-y-1">
                {credits.map(([slug, credit]) => (
                  <li key={slug}>
                    <a
                      href={credit.source}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="underline underline-offset-2"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {credit.player}
                    </a>
                    {" — © "}
                    {credit.author}
                    {", "}
                    <a
                      href={credit.licence_url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="underline underline-offset-2"
                    >
                      {credit.licence}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {placeholders.length > 0 && (
            <div>
              <p className="mb-1.5" style={{ color: "var(--text-secondary)" }}>
                Shown with the generic avatar — no photo could be confirmed under a free
                licence:
              </p>
              <ul className="space-y-1">
                {placeholders.map((entry) => (
                  <li key={entry.slug}>
                    {entry.player} — {entry.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
