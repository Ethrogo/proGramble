import Link from "next/link";

import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";

const sports = [
  { href: "/mlb", label: "MLB", note: "Daily pitcher props, featured matchups, and baseball boards built for fast reads." },
  { href: "/nba", label: "NBA", note: "Game-night player markets and star-driven slate browsing that keeps the focus on the plays people care about." },
  { href: "/nfl", label: "NFL", note: "Weekly boards for marquee matchups, player props, and weekend planning." },
  { href: "/tennis", label: "Tennis", note: "ATP and WTA match pages designed around the players, props, and moments fans want first." },
  { href: "/golf", label: "Golf", note: "Tournament and round boards that make golfer matchups and prop angles easier to follow." }
];

const featuredPaths = [
  {
    href: "/mlb",
    title: "Start with MLB",
    description: "Jump into pitcher-focused baseball pages built around today's games, featured arms, and popular props."
  },
  {
    href: "/nba",
    title: "Browse NBA boards",
    description: "See how ProGramble is shaping a faster game-night experience for player props and headline matchups."
  },
  {
    href: "/golf",
    title: "Preview golf coverage",
    description: "Explore how tournament pages, round angles, and golfer matchups fit into the same clean experience."
  }
];

const productPillars = [
  {
    title: "See the board faster",
    body: "ProGramble is built to help users get from homepage to the most interesting games and props without digging through clutter."
  },
  {
    title: "Start with popular props",
    body: "Each sport page is designed around the markets fans naturally look for first, from pitcher strikeouts to star-player props."
  },
  {
    title: "One brand across sports",
    body: "Whether the sport is daily, weekly, match-based, or tournament-based, the experience stays familiar and easy to revisit."
  }
];

const userBenefits = [
  "Cleaner daily browsing across major sports",
  "Faster access to featured games and props",
  "A more welcoming entry point for repeat visits"
];

const whyNow = [
  "Follow the sports you already care about in one place",
  "Build a daily habit around featured slates and popular props",
  "Get familiar with the experience before deeper pages roll out"
];

export default function HomePage() {
  return (
    <>
      <PageHero
        eyebrow="ProGramble"
        title="Find the games and props worth your attention faster."
        description="ProGramble is shaping a cleaner sports experience around featured slates, popular player props, and the matchups fans want to check first across MLB, NBA, NFL, tennis, and golf."
        actions={[
          { href: "/mlb", label: "Explore MLB", variant: "primary" },
          { href: "/about", label: "Why ProGramble", variant: "secondary" }
        ]}
        sideTitle="Built for repeat visits"
        sideDescription="From first glance to final card, the site is meant to help users move quickly, stay oriented, and keep coming back for the next board."
        stats={[
          { label: "Sports in focus", value: "5" },
          { label: "Best for", value: "Featured props" },
          { label: "Experience", value: "Fast daily browsing" }
        ]}
      />

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">Why users will care</p>
          <h2 className="section-title">A sharper, simpler way to browse sports props.</h2>
        </div>
        <div className="section-grid">
          {productPillars.map((pillar) => (
            <SectionCard key={pillar.title} title={pillar.title}>
              <p>{pillar.body}</p>
              <div className="pill-row">
                <span className="pill">Cleaner boards</span>
                <span className="pill">Less clutter</span>
              </div>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">Pick your sport</p>
          <h2 className="section-title">Start where you already spend your attention.</h2>
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
                  Browse {sport.label}
                </Link>
              </p>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>Where to start</h3>
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
          <h3>What makes the experience better</h3>
          <ul className="simple-list">
            <li>Featured pages keep the focus on the best boards instead of endless tables.</li>
            <li>Prop discovery feels faster when the important games and players are easier to find.</li>
            <li>The same clear experience carries from daily baseball to weekly football and beyond.</li>
          </ul>
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>What new users can expect</h3>
          <ul className="feature-list">
            {userBenefits.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="info-list">
          <h3>Why start now</h3>
          <ul className="simple-list">
            {whyNow.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>
    </>
  );
}
