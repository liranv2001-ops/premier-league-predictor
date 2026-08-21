import type { TeamRow } from "../lib/predictions";
import { positionInterval, RELEGATION_FROM, UCL_PLACES } from "../lib/predictions";
import PositionRange from "./PositionRange";
import ProbabilityBar from "./ProbabilityBar";
import TeamBadge from "./TeamBadge";

interface Props {
  table: TeamRow[];
}

type Zone = "ucl" | "relegation" | null;

function zoneOf(rank: number): Zone {
  if (rank <= UCL_PLACES) return "ucl";
  if (rank >= RELEGATION_FROM) return "relegation";
  return null;
}

/**
 * Zone marker: a coloured rule *and* a text label, never colour alone.
 *
 * Only relegation gets a colour, and it is the reserved critical red. The Champions
 * League places are marked with a neutral rule and the "UCL" label instead of green -
 * red against green measures ΔE 4.1 under deuteranopia, so a red/green reader could not
 * separate the top of the table from the bottom. The label does the work in both cases.
 */
function zoneRule(zone: Zone): string {
  if (zone === "relegation") return "var(--critical)";
  if (zone === "ucl") return "var(--series-1)";
  return "transparent";
}

function zoneLabel(zone: Zone): string | null {
  if (zone === "relegation") return "REL";
  if (zone === "ucl") return "UCL";
  return null;
}

/**
 * The predicted table.
 *
 * Twenty clubs all carrying meaning is well past the point where colour can separate
 * them, so this is a table rather than a chart - which also gives every chart on the
 * page its accessible table-view twin for free.
 */
export default function LeagueTable({ table }: Props) {
  return (
    <section
      className="rounded-2xl"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-card)",
      }}
      aria-labelledby="table-heading"
    >
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 px-5 pt-5 sm:px-6">
        <h2 id="table-heading" className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
          Predicted final table
        </h2>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Bands show the 80% range of finishing positions across 10,000 simulated seasons.
        </p>
      </div>

      {/* Narrow screens get stacked rows instead of a five-column table. Sideways
          scrolling through twenty clubs is a worse read than dropping the one column
          that genuinely needs width - the finish range, which is meaningless squeezed
          into 60px. Rank, club, points and title chance all survive. */}
      <ul className="mt-3 divide-y sm:hidden" style={{ borderColor: "var(--gridline)" }}>
        {table.map((row) => {
          const zone = zoneOf(row.predicted_rank);
          const label = zoneLabel(zone);
          const interval = positionInterval(row.position_distribution);
          return (
            <li
              key={row.slug}
              className="relative flex items-center gap-3 py-3 pr-5 pl-5"
              style={{ borderColor: "var(--gridline)" }}
            >
              <span
                aria-hidden="true"
                className="absolute inset-y-0 left-0 w-[3px]"
                style={{ background: zoneRule(zone) }}
              />
              <span
                className="tabular w-5 shrink-0 text-right font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                {row.predicted_rank}
              </span>
              <TeamBadge slug={row.slug} name={row.team} size={26} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <span
                    className="truncate text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {row.team}
                  </span>
                  {label && (
                    <span
                      className="shrink-0 rounded px-1 text-[10px] font-semibold"
                      style={{
                        color: zone === "relegation" ? "var(--critical)" : "var(--series-1)",
                        border: `1px solid ${zoneRule(zone)}`,
                      }}
                    >
                      {label}
                    </span>
                  )}
                </div>
                <p className="tabular mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
                  {row.expected_points.toFixed(1)} pts · finish {interval.low}–{interval.high}
                </p>
              </div>
              <div className="w-24 shrink-0">
                <ProbabilityBar value={row.title_probability} />
              </div>
            </li>
          );
        })}
      </ul>

      {/* Wide content scrolls inside its own box; the page never scrolls sideways. */}
      <div className="mt-4 hidden overflow-x-auto sm:block">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <thead>
            <tr
              className="text-xs tracking-wide uppercase"
              style={{ color: "var(--text-muted)" }}
            >
              <th scope="col" className="w-12 px-2 py-2 text-right font-medium">
                #
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Club
              </th>
              <th scope="col" className="w-28 px-3 py-2 text-right font-medium">
                Points
              </th>
              <th scope="col" className="w-48 px-3 py-2 text-left font-medium">
                Title chance
              </th>
              <th scope="col" className="w-56 px-3 py-2 text-left font-medium">
                Finish range
              </th>
            </tr>
          </thead>
          <tbody>
            {table.map((row) => {
              const zone = zoneOf(row.predicted_rank);
              const label = zoneLabel(zone);
              return (
                <tr
                  key={row.slug}
                  className="border-t transition-colors"
                  style={{ borderColor: "var(--gridline)" }}
                >
                  <td className="relative px-2 py-2.5 text-right">
                    <span
                      aria-hidden="true"
                      className="absolute inset-y-0 left-0 w-[3px]"
                      style={{ background: zoneRule(zone) }}
                    />
                    <span className="tabular font-semibold" style={{ color: "var(--text-primary)" }}>
                      {row.predicted_rank}
                    </span>
                  </td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2.5">
                      <TeamBadge slug={row.slug} name={row.team} size={24} />
                      <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                        {row.team}
                      </span>
                      {label && (
                        <span
                          className="rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide"
                          style={{
                            color: zone === "relegation" ? "var(--critical)" : "var(--series-1)",
                            border: `1px solid ${zoneRule(zone)}`,
                          }}
                        >
                          {label}
                        </span>
                      )}
                    </div>
                  </td>
                  <td
                    className="tabular px-3 py-2.5 text-right"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {row.expected_points.toFixed(1)}
                  </td>
                  <td className="px-3 py-2.5">
                    <ProbabilityBar value={row.title_probability} />
                  </td>
                  <td className="px-3 py-2.5">
                    <PositionRange
                      interval={positionInterval(row.position_distribution)}
                      team={row.team}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div
        className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t px-5 py-3 text-xs sm:px-6"
        style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
      >
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-[3px]" style={{ background: "var(--series-1)" }} />
          UCL — Champions League places
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-[3px]" style={{ background: "var(--critical)" }} />
          REL — relegation places
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="h-1.5 w-6 rounded-full"
            style={{ background: "var(--series-1-soft)" }}
          />
          80% finish range
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ background: "var(--series-1)" }}
          />
          most likely finish
        </span>
      </div>
    </section>
  );
}
