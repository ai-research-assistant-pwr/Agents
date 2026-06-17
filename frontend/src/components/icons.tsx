export function LogoIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <rect x="0.75" y="0.75" width="16.5" height="16.5" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <rect x="3" y="3" width="3" height="3" fill="currentColor" />
      <rect x="12" y="12" width="3" height="3" fill="currentColor" />
      <line x1="6" y1="12" x2="12" y2="6" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function PlusIcon() {
  return <span className="plus-glyph" aria-hidden="true">+</span>;
}

export function SearchIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <circle cx="5" cy="5" r="3.5" stroke="currentColor" strokeWidth="1.4" />
      <line x1="7.5" y1="7.5" x2="11" y2="11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function AgentMark() {
  return (
    <svg width="10" height="10" viewBox="-1 -1 12 12" fill="none" aria-hidden="true">
      <rect x="1.5" y="1.5" width="7" height="7" stroke="currentColor" strokeWidth="1.5" transform="rotate(45 5 5)" />
    </svg>
  );
}

export function ArrowUpRight() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
      <path d="M2.5 7.5L7.5 2.5M3 2.5h4.5V7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function GraphIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden="true">
      <circle cx="2.5" cy="2.5" r="1.5" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="8.5" cy="2.5" r="1.5" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="5.5" cy="8.5" r="1.5" stroke="currentColor" strokeWidth="1.2" />
      <line x1="3.5" y1="3.5" x2="5" y2="7" stroke="currentColor" strokeWidth="1.2" />
      <line x1="7.5" y1="3.5" x2="6" y2="7" stroke="currentColor" strokeWidth="1.2" />
      <line x1="4" y1="2.5" x2="7" y2="2.5" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
