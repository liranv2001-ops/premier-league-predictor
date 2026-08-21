import { useState } from "react";
import { PLACEHOLDER_SRC, photoSrc, type PlayerMapping } from "../lib/assets";
import { initials, slugHue } from "../lib/predictions";

interface Props {
  slug: string;
  name: string;
  mapping?: PlayerMapping | null;
  size?: number;
}

/**
 * A player photo, degrading through two fallbacks.
 *
 * 1. The free-licensed photograph from Wikimedia Commons, if one was verified.
 * 2. The project's generic placeholder avatar, for anyone on the mapping's
 *    `placeholders` list — a player whose photo could not be confirmed as *them* under
 *    a free licence.
 * 3. An initials monogram, if even the placeholder is absent (a fresh clone that has
 *    not run the asset pipeline).
 *
 * The second step is the one that matters: it is better to show a deliberate anonymous
 * avatar than either a stranger's face or a broken image.
 */
export default function PlayerPhoto({ slug, name, mapping, size = 96 }: Props) {
  const [stage, setStage] = useState<"photo" | "placeholder" | "monogram">("photo");
  const hue = slugHue(slug);

  const hasVerifiedPhoto = Boolean(mapping?.players[slug]);
  const src = stage === "photo" && hasVerifiedPhoto ? photoSrc(slug, mapping ?? null) : PLACEHOLDER_SRC;

  const advance = () => setStage(stage === "photo" ? "placeholder" : "monogram");

  return (
    <div
      className="relative shrink-0 overflow-hidden rounded-full"
      style={{
        width: size,
        height: size,
        background: `hsl(${hue} 30% 90%)`,
        border: "1px solid var(--border)",
      }}
    >
      {stage === "monogram" ? (
        <span
          aria-hidden="true"
          className="flex h-full w-full items-center justify-center font-semibold"
          style={{ fontSize: size * 0.34, color: `hsl(${hue} 45% 30%)` }}
        >
          {initials(name)}
        </span>
      ) : (
        <img
          src={src}
          alt={hasVerifiedPhoto && stage === "photo" ? name : `${name} (no free photo available)`}
          width={size}
          height={size}
          loading="lazy"
          onError={advance}
          className="h-full w-full object-cover object-top"
        />
      )}
    </div>
  );
}
