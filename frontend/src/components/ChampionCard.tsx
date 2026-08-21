import type { Predictions } from "../lib/predictions";
import { percent } from "../lib/predictions";
import TeamBadge from "./TeamBadge";

interface Props {
  predictions: Predictions;
}

/**
 * The predicted champion, as a hero figure.
 *
 * One number is the point, so this is a hero figure rather than a chart - a one-bar bar
 * chart is a named anti-pattern. The figure stays in the system sans with proportional
 * figures; tabular-nums on a display-size number makes it look loose.
 */
export default function ChampionCard({ predictions }: Props) {
  const { champion, table } = predictions;
  const row = table.find((entry) => entry.slug === champion.slug);
  const runnerUp = table[1];

  return (
    <section
      className="rounded-2xl p-6 sm:p-8"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-card)",
      }}
      aria-labelledby="champion-heading"
    >
      <p
        id="champion-heading"
        className="text-xs font-semibold tracking-[0.14em] uppercase"
        style={{ color: "var(--text-muted)" }}
      >
        Predicted champion · {predictions.season}
      </p>

      <div className="mt-5 flex flex-col gap-6 sm:flex-row sm:items-center sm:gap-8">
        <div className="flex items-center gap-4">
          <TeamBadge slug={champion.slug} name={champion.team} size={72} />
          <div>
            <h2 className="text-2xl font-bold sm:text-3xl" style={{ color: "var(--text-primary)" }}>
              {champion.team}
            </h2>
            {row && (
              <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
                {row.expected_points.toFixed(1)} expected points
              </p>
            )}
          </div>
        </div>

        <div className="sm:ml-auto sm:text-right">
          <div
            className="hero-figure text-5xl leading-none font-bold sm:text-6xl"
            style={{ color: "var(--series-1)" }}
          >
            {percent(champion.probability, 1)}
          </div>
          <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
            chance of the title
          </p>
        </div>
      </div>

      {runnerUp && (
        <p
          className="mt-6 border-t pt-4 text-sm"
          style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
        >
          Closest challenger:{" "}
          <span style={{ color: "var(--text-primary)" }}>{runnerUp.team}</span> at{" "}
          <span className="tabular">{percent(runnerUp.title_probability, 1)}</span>
        </p>
      )}
    </section>
  );
}
