import { percent } from "../lib/predictions";

interface Props {
  value: number;
  label?: string;
}

/**
 * A single-hue meter for a probability.
 *
 * Every bar is the same colour. Shading each one darker-where-bigger would double-encode
 * length as hue and burn the only free channel on information the length already carries
 * - it is a named anti-pattern, and clubs have no natural order to ramp along anyway.
 *
 * The value is always present as text beside the bar, so the mark enhances rather than
 * gates: nothing here is readable only by eye.
 */
export default function ProbabilityBar({ value, label }: Props) {
  const width = Math.max(value * 100, value > 0 ? 1.5 : 0);

  return (
    <div className="flex items-center gap-2">
      <div
        className="relative h-1.5 w-full min-w-10 overflow-hidden rounded-full"
        style={{ background: "var(--gridline)" }}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${width}%`, background: "var(--series-1)" }}
        />
      </div>
      <span
        className="tabular w-12 shrink-0 text-right text-xs"
        style={{ color: value > 0.005 ? "var(--text-primary)" : "var(--text-muted)" }}
      >
        {label ?? percent(value, value >= 0.1 ? 0 : 1)}
      </span>
    </div>
  );
}
