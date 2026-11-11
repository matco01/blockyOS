/** The Blocky mark: four colored blocks. */
export default function Logo({ size = 26, animated = false }: { size?: number; animated?: boolean }) {
  return (
    <svg className={`logo ${animated ? "animated" : ""}`} width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="1.5" y="1.5" width="9.5" height="9.5" rx="3" fill="#7c5cff" />
      <rect x="13" y="1.5" width="9.5" height="9.5" rx="3" fill="#ff5c9d" />
      <rect x="1.5" y="13" width="9.5" height="9.5" rx="3" fill="#ffa94d" />
      <rect x="13" y="13" width="9.5" height="9.5" rx="3" fill="#22d3ee" />
    </svg>
  );
}
