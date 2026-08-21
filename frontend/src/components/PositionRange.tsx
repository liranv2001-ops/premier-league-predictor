import { LEAGUE_SIZE, type Interval } from "../lib/predictions";

interface Props {
  interval: Interval;
  team: string;
}

/**
 * The 80% credible interval of a club's finishing position, on a shared 1-20 axis.
 *
 * Every row uses the same scale, so the bands are comparable down the column: a club
 * spanning 1st-9th is visibly less certain than one spanning 1st-3rd. This is the actual
 * spread of 10,000 simulated seasons, not a progress bar dressed up as uncertainty.
 *
 * The numeric range is rendered as text too - the band is the quick read, the text is
 * the exact one.
 */
export default function PositionRange({ interval, team }: Props) {
  const { low, high, median } = interval;

  // Positions are discrete, so the band spans from the *start* of `low` to the *end*
  // of `high`; treating them as points would make a one-place band invisible.
  const left = ((low - 1) / LEAGUE_SIZE) * 100;
  const right = (high / LEAGUE_SIZE) * 100;
  const medianLeft = ((median - 0.5) / LEAGUE_SIZE) * 100;

  const description =
    low === high
      ? `${team}: 80% chance of finishing ${low}`
      : `${team}: 80% chance of finishing between ${low} and ${high}, most likely ${median}`;

  return (
    <div className="flex items-center gap-2.5">
      <div
        className="relative h-6 w-full min-w-24"
        role="img"
        aria-label={description}
        title={description}
      >
        {/* Recessive track: a solid hairline, never dashed. */}
        <div
          className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2"
          style={{ background: "var(--gridline)" }}
        />
        <div
          className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full"
          style={{
            left: `${left}%`,
            width: `${Math.max(right - left, 2)}%`,
            background: "var(--series-1-soft)",
          }}
        />
        {/* The median sits on top with a 2px surface ring, so it stays legible where it
            overlaps the band rather than needing a border drawn around it. */}
        <div
          className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            left: `${medianLeft}%`,
            background: "var(--series-1)",
            boxShadow: "0 0 0 2px var(--surface-1)",
          }}
        />
      </div>
      <span
        className="tabular w-11 shrink-0 text-right text-xs"
        style={{ color: "var(--text-secondary)" }}
      >
        {low === high ? low : `${low}–${high}`}
      </span>
    </div>
  );
}
