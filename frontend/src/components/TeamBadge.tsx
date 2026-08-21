import { useState } from "react";
import { initials, slugHue } from "../lib/predictions";

interface Props {
  slug: string;
  name: string;
  size?: number;
}

/**
 * A club badge, with a monogram fallback that always renders.
 *
 * The badges are SVGs this project generates in each club's colours — **not** official
 * crests. Premier League crests are trademarked and are not available under an open
 * licence, so drawing our own is the only honest way to ship "open-licensed logos".
 *
 * The runtime monogram stays as a final safety net: badges are git-ignored, so a fresh
 * clone that has not run the generator must still render a clean table rather than
 * twenty broken-image icons.
 */
export default function TeamBadge({ slug, name, size = 28 }: Props) {
  const [failed, setFailed] = useState(false);
  const hue = slugHue(slug);

  if (failed) {
    return (
      <span
        aria-hidden="true"
        className="inline-flex shrink-0 items-center justify-center rounded-full font-semibold"
        style={{
          width: size,
          height: size,
          fontSize: size * 0.38,
          background: `hsl(${hue} 42% 88%)`,
          color: `hsl(${hue} 55% 26%)`,
          border: "1px solid var(--border)",
        }}
      >
        {initials(name)}
      </span>
    );
  }

  return (
    <img
      src={`${import.meta.env.BASE_URL}logos/${slug}.svg`}
      alt=""
      aria-hidden="true"
      width={size}
      height={size}
      loading="lazy"
      onError={() => setFailed(true)}
      className="shrink-0 object-contain"
      style={{ width: size, height: size }}
    />
  );
}
