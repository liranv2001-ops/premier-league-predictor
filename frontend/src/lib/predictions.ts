/**
 * Types and helpers for the prediction payload.
 *
 * The shape mirrors `data/processed/predictions.json`, produced by
 * `src/models/predictions.py`. A Python test asserts the payload carries every field
 * read here, so a schema change breaks a test rather than the page.
 */

export interface TeamRow {
  team: string;
  slug: string;
  predicted_position: number;
  predicted_rank: number;
  title_probability: number;
  expected_points: number;
  position_distribution: Record<string, number>;
}

export interface AwardCandidate {
  player: string;
  slug: string;
  team: string;
  team_slug: string;
  probability?: number;
  score?: number;
  predicted_goals?: number;
  predicted_assists?: number;
  predicted_minutes?: number;
  components?: { attacking: number; team: number; minutes: number };
}

export interface AwardBlock extends AwardCandidate {
  candidates: AwardCandidate[];
}

export interface Predictions {
  generated_at: string;
  season: string;
  model_version: string;
  cutoff_matchweek: number;
  n_simulations: number;
  assumptions: string[];
  validation: Record<string, number>;
  table: TeamRow[];
  champion: { team: string; slug: string; probability: number };
  top_scorer: AwardBlock;
  top_assists: AwardBlock;
  player_of_the_season: AwardBlock;
}

export const LEAGUE_SIZE = 20;

/** Clubs finishing here qualify for the Champions League. */
export const UCL_PLACES = 4;

/** Clubs finishing at or below this go down. */
export const RELEGATION_FROM = 18;

export interface Interval {
  low: number;
  high: number;
  median: number;
}

/**
 * The 80% credible interval of a club's finishing position.
 *
 * Walks the cumulative distribution to the 10th, 50th and 90th percentiles. This is
 * real uncertainty straight from the Monte Carlo, not a decorative bar - a club whose
 * band spans 1st to 9th is genuinely less certain than one spanning 1st to 3rd.
 */
export function positionInterval(distribution: Record<string, number>): Interval {
  let cumulative = 0;
  let low = 1;
  let median = 1;
  let high = LEAGUE_SIZE;
  let seenLow = false;
  let seenMedian = false;

  for (let position = 1; position <= LEAGUE_SIZE; position += 1) {
    cumulative += distribution[String(position)] ?? 0;
    if (!seenLow && cumulative >= 0.1) {
      low = position;
      seenLow = true;
    }
    if (!seenMedian && cumulative >= 0.5) {
      median = position;
      seenMedian = true;
    }
    if (cumulative >= 0.9) {
      high = position;
      break;
    }
  }
  return { low, high, median };
}

/** Probability a club finishes in the relegation places. */
export function relegationRisk(distribution: Record<string, number>): number {
  let total = 0;
  for (let position = RELEGATION_FROM; position <= LEAGUE_SIZE; position += 1) {
    total += distribution[String(position)] ?? 0;
  }
  return total;
}

/** Format a probability as a percentage string. */
export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Initials for a fallback badge, at most two characters. */
export function initials(name: string): string {
  const words = name.split(/[\s-]+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}

/**
 * A stable hue per club, used only for the monogram fallback.
 *
 * This is decoration on a placeholder, never a data encoding - nothing in the charts
 * is coloured by club.
 */
export function slugHue(slug: string): number {
  let hash = 0;
  for (let i = 0; i < slug.length; i += 1) {
    hash = (hash * 31 + slug.charCodeAt(i)) % 360;
  }
  return hash;
}

/** Load the payload Vite serves from `public/`. */
export async function loadPredictions(): Promise<Predictions> {
  const response = await fetch(`${import.meta.env.BASE_URL}predictions.json`);
  if (!response.ok) {
    throw new Error(
      `Could not load predictions.json (${response.status}). ` +
        `Run: python -m src.models.cli --season 2026/27 ` +
        `--teams-file data/seasons/2026-27.txt --awards --publish`,
    );
  }
  return (await response.json()) as Predictions;
}
