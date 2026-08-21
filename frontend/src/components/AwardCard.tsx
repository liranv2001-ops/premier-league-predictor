import type { PlayerMapping } from "../lib/assets";
import type { AwardBlock, AwardCandidate } from "../lib/predictions";
import { percent } from "../lib/predictions";
import PlayerPhoto from "./PlayerPhoto";
import TeamBadge from "./TeamBadge";

interface Props {
  title: string;
  award: AwardBlock;
  /** The headline statistic for the leader, e.g. "15.7 goals". */
  stat: (candidate: AwardCandidate) => string;
  /** How each shortlist row is scored, shown on the right. */
  secondary: (candidate: AwardCandidate) => string;
  /** Photo credits, so the card knows whether a verified photo exists. */
  photos?: PlayerMapping | null;
}

/**
 * One individual award: the favourite, then the rest of the shortlist.
 *
 * The shortlist matters as much as the leader - a 63% favourite and a 14% favourite look
 * identical if you only show the winner, and these probabilities differ by that much
 * between the awards.
 */
export default function AwardCard({ title, award, stat, secondary, photos }: Props) {
  const [leader, ...rest] = award.candidates;
  if (!leader) return null;

  return (
    <section
      className="flex flex-col rounded-2xl p-5"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-card)",
      }}
      aria-label={title}
    >
      <p
        className="text-xs font-semibold tracking-[0.14em] uppercase"
        style={{ color: "var(--text-muted)" }}
      >
        {title}
      </p>

      <div className="mt-4 flex items-center gap-4">
        <PlayerPhoto slug={leader.slug} name={leader.player} mapping={photos} size={72} />
        <div className="min-w-0">
          <h3
            className="truncate text-lg font-bold"
            style={{ color: "var(--text-primary)" }}
            title={leader.player}
          >
            {leader.player}
          </h3>
          <div className="mt-1 flex items-center gap-1.5">
            <TeamBadge slug={leader.team_slug} name={leader.team} size={16} />
            <span className="truncate text-sm" style={{ color: "var(--text-secondary)" }}>
              {leader.team}
            </span>
          </div>
          <p className="hero-figure mt-2 text-xl font-semibold" style={{ color: "var(--series-1)" }}>
            {stat(leader)}
          </p>
        </div>
      </div>

      {rest.length > 0 && (
        <ol className="mt-5 space-y-2 border-t pt-4" style={{ borderColor: "var(--border)" }}>
          {rest.map((candidate, index) => (
            <li key={candidate.slug} className="flex items-center gap-2 text-sm">
              <span
                className="tabular w-4 shrink-0 text-right text-xs"
                style={{ color: "var(--text-muted)" }}
              >
                {index + 2}
              </span>
              <TeamBadge slug={candidate.team_slug} name={candidate.team} size={16} />
              <span className="truncate" style={{ color: "var(--text-primary)" }}>
                {candidate.player}
              </span>
              <span
                className="tabular ml-auto shrink-0 text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                {secondary(candidate)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

/** Format helpers, kept beside the card that uses them. */
export const formatters = {
  goals: (c: AwardCandidate) => `${(c.predicted_goals ?? 0).toFixed(1)} goals`,
  assists: (c: AwardCandidate) => `${(c.predicted_assists ?? 0).toFixed(1)} assists`,
  score: (c: AwardCandidate) => `${((c.score ?? 0) * 100).toFixed(0)} rating`,
  probability: (c: AwardCandidate) => percent(c.probability ?? 0, 1),
  ratingValue: (c: AwardCandidate) => ((c.score ?? 0) * 100).toFixed(0),
};
