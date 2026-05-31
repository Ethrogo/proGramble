import Link from "next/link";

import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";

const sports = [
  { href: "/mlb", label: "MLB", note: "Pitcher props, live slates, and yesterday's tracked strikeout and walk results." },
  { href: "/nba", label: "NBA", note: "Lineup-sensitive player markets and fast slate views for game day." },
  { href: "/nfl", label: "NFL", note: "Weekly game boards, player markets, and injury-aware context." },
  { href: "/tennis", label: "Tennis", note: "ATP and WTA match pages with player-focused market structure." },
  { href: "/golf", label: "Golf", note: "PGA and LPGA tournament, round, and matchup market shells." }
];

const featuredPaths = [
  {
    href: "/mlb",
    title: "Open today's MLB slate",
    description: "Start with pitcher-focused MLB pages, then move into player prop detail and yesterday's tracked results."
  },
  {
    href: "/tennis",
    title: "Browse ATP/WTA markets",
    description: "Individual-sport routing is already shaped for match slates, player pages, and competition-specific context."
  },
  {
    href: "/golf",
    title: "Review featured golf boards",
    description: "Tournament and round structures are part of the same navigation system as team-sport slates."
  }
];

const productPillars = [
  {
    title: "Live slates",
    body: "The site surfaces sport-level slates first, then lets users drill into games, matches, or rounds without exposing raw provider payloads."
  },
  {
    title: "Featured markets",
    body: "Each sport route is designed to promote the highest-signal player and event markets instead of dumping undifferentiated odds tables."
  },
  {
    title: "Tracked results",
    body: "MLB starts with yesterday's `pitcher_k` and `pitcher_bb` outcomes sourced from internal tracking data and served through the website API."
  }
];

export default function HomePage() {
  return (
    <>
      <PageHero
        eyebrow="ProGramble"
        title="A multi-sport front door for slates, player props, and tracked betting results."
        description="ProGramble is building a clean public surface for sports betting research across MLB, NBA, NFL, ATP, WTA, PGA, and LPGA. The website routes users into live slates, featured player markets, and yesterday's modeled results without coupling the browser to internal odds or model pipelines."
        actions={[
          { href: "/mlb", label: "Open live MLB slate", variant: "primary" },
          { href: "/about", label: "View product and stack notes", variant: "secondary" }
        ]}
        sideTitle="Current MVP focus"
        sideDescription="The initial release centers on public discovery pages, sport-level navigation, featured market entry points, and a backend bridge for yesterday's MLB pitcher results."
        stats={[
          { label: "Supported routes", value: "Home + 6 sport pages" },
          { label: "Primary backend", value: "Spring Boot API" },
          { label: "Tracked results", value: "pitcher_k and pitcher_bb" }
        ]}
      />

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">What the site does</p>
          <h2 className="section-title">Public product pages first, deeper data behind the API.</h2>
        </div>
        <div className="section-grid">
          {productPillars.map((pillar) => (
            <SectionCard key={pillar.title} title={pillar.title}>
              <p>{pillar.body}</p>
              <div className="pill-row">
                <span className="pill">SEO-friendly pages</span>
                <span className="pill">Internal API only</span>
              </div>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">Supported sports</p>
          <h2 className="section-title">One navigation system across team sports and individual sports.</h2>
        </div>
        <div className="section-grid">
          {sports.map((sport) => (
            <SectionCard key={sport.href} title={sport.label}>
              <p>{sport.note}</p>
              <div className="pill-row">
                <span className="pill">Landing page</span>
                <span className="pill">Slate path</span>
                <span className="pill">Featured markets</span>
              </div>
              <p>
                <Link href={sport.href} className="mini-link">
                  Explore {sport.label}
                </Link>
              </p>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>Featured launch paths</h3>
          <div className="feature-stack">
            {featuredPaths.map((path) => (
              <div key={path.href} className="feature-row">
                <div>
                  <strong>{path.title}</strong>
                  <p>{path.description}</p>
                </div>
                <Link href={path.href} className="mini-link">
                  Open
                </Link>
              </div>
            ))}
          </div>
        </div>
        <div className="info-list">
          <h3>Why the backend boundary matters</h3>
          <ul className="simple-list">
            <li>Frontend routes stay stable even as odds providers, models, and grading pipelines change.</li>
            <li>Yesterday's results can come from tracked internal artifacts without teaching the browser about file layout.</li>
            <li>Future live slates and featured markets can share one API namespace across every supported sport.</li>
          </ul>
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>Immediate MVP deliverables</h3>
          <ul className="feature-list">
            <li>Explain the product clearly on the homepage.</li>
            <li>Route users into sports, slates, and player prop pages.</li>
            <li>Expose MLB pitcher strikeout and walk results from yesterday through the API layer.</li>
          </ul>
        </div>
        <div className="info-list">
          <h3>Next implementation targets</h3>
          <ul className="simple-list">
            <li>Add `/api/v1/sports`, `/api/v1/events`, and `/api/v1/results/yesterday`.</li>
            <li>Back the API with Postgres using the initial cross-sport schema.</li>
            <li>Introduce admin refresh pages once auth is in place.</li>
          </ul>
        </div>
      </section>
    </>
  );
}
