import Link from "next/link";
import { ReactNode } from "react";

import { NavLink } from "./site-nav-link";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/mlb", label: "MLB" },
  { href: "/nba", label: "NBA" },
  { href: "/nfl", label: "NFL" },
  { href: "/tennis", label: "Tennis" },
  { href: "/golf", label: "Golf" },
  { href: "/about", label: "Why ProGramble" }
];

export function SiteChrome({ children }: { children: ReactNode }) {
  return (
    <div className="site-shell">
      <div className="topband">Featured slates, popular props, and cleaner game-day browsing</div>
      <header className="site-header">
        <div className="site-header-inner">
          <Link href="/" className="brand" aria-label="ProGramble home">
            <span className="brand-mark">ProGramble</span>
            <span className="brand-copy">A simpler way to discover the games, props, and storylines worth your attention</span>
          </Link>
          <nav className="site-nav" aria-label="Primary">
            {navItems.map((item) => (
              <NavLink key={item.href} href={item.href}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="page-wrap">{children}</main>

      <footer className="site-footer">
        <div className="site-footer-inner">
        <div>
          <strong>ProGramble</strong>
          <p>
              Built for sports fans who want faster access to featured boards, player props, and the moments that shape the day.
          </p>
        </div>
        <div className="footer-links">
          <Link href="/mlb">MLB</Link>
          <Link href="/nba">NBA</Link>
          <Link href="/nfl">NFL</Link>
          <Link href="/tennis">Tennis</Link>
          <Link href="/golf">Golf</Link>
            <Link href="/about">Why ProGramble</Link>
        </div>
      </div>
    </footer>
  </div>
  );
}
