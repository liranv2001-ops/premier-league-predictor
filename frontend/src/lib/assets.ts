/**
 * Asset mappings: where images live and who to credit for them.
 *
 * `players-mapping.json` is the attribution record. CC BY and CC BY-SA require credit,
 * so the licence, author and source travel with every photo and are displayed in the
 * dashboard footer. A mapping file nobody sees is not attribution.
 */

export interface PhotoCredit {
  player: string;
  club: string;
  path: string;
  licence: string;
  licence_url: string;
  author: string;
  source: string;
  article: string;
}

export interface PlaceholderEntry {
  player: string;
  slug: string;
  club?: string;
  path: string;
  reason: string;
}

export interface PlayerMapping {
  generated_at: string;
  note: string;
  players: Record<string, PhotoCredit>;
  placeholders: PlaceholderEntry[];
}

export interface LogoMapping {
  generated_at: string;
  note: string;
  licence: string;
  logos: Record<string, { team: string; path: string; initials: string; colours: string[] }>;
}

export const PLACEHOLDER_SRC = `${import.meta.env.BASE_URL}players/_placeholder.svg`;

/** Which file extension a player's photo was saved with, defaulting to jpg. */
export function photoSrc(slug: string, mapping: PlayerMapping | null): string {
  const entry = mapping?.players[slug];
  if (!entry) return PLACEHOLDER_SRC;
  const filename = entry.path.split("/").pop() ?? `${slug}.jpg`;
  return `${import.meta.env.BASE_URL}players/${filename}`;
}

async function loadJson<T>(name: string): Promise<T | null> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}${name}`);
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    // The dashboard must work before the asset pipeline has ever run.
    return null;
  }
}

export function loadPlayerMapping(): Promise<PlayerMapping | null> {
  return loadJson<PlayerMapping>("players-mapping.json");
}

export function loadLogoMapping(): Promise<LogoMapping | null> {
  return loadJson<LogoMapping>("logos-mapping.json");
}
