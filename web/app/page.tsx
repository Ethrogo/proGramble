import Link from "next/link";

import { PageHero } from "../components/page-hero";
import { SectionCard } from "../components/section-card";

const sports = [
  { href: "/mlb", label: "MLB", note: "Pitcher props, slates, and tracked results" },
  { href: "/nba", label: "NBA", note: "Lineup-sensitive player market shell" },
  { href: "/nfl", label: "NFL", note: "Weekly boards and player props" },
  { href: "/tennis", label: "Tennis", note: "ATP/WTA match and player market flow" },
  { href: "/golf", label: "Golf", note: "PGA/LPGA tournament and round views" }
];

export default function HomePage() {
  return (
    <>
      <PageHero
        eyebrow="Website MVP"
        title="A multi-sport shell for slates, props, tracked results, and admin workflows."
        description="This frontend is the public surface described in the MADR: a Next.js website shell backed by a Spring Boot API for sports, events, offers, yesterday's results, and admin refresh operations."
        actions={[
          { href: "/mlb", label: "Open MLB shell", variant: "primary" },
          { href: "/about", label: "View stack and boundaries", variant: "secondary" }
        ]}
        sideTitle="Initial page system"
        sideDescription="The shell is organized around public discovery pages now, with backend-driven data and admin controls added through authenticated API routes later."
        stats={[
          { label: "Public routes", value: "7 base routes" },
          { label: "Primary backend", value: "Spring Boot API" },
          { label: "Results bridge", value: "pitcher_k / pitcher_bb" }
        ]}
      />

      <section className="page-block">
        <div className="section-grid">
          {sports.map((sport) => (
            <SectionCard key={sport.href} title={sport.label}>
              <p>{sport.note}</p>
              <div className="pill-row">
                <span className="pill">Landing page</span>
                <span className="pill">Slate path</span>
                <span className="pill">Prop detail path</span>
              </div>
              <p>
                <Link href={sport.href} className="mini-link">
                  Open {sport.label}
                </Link>
              </p>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>Shell responsibilities</h3>
          <ul className="feature-list">
            <li>Present SEO-friendly public pages for sports, slates, and prop discovery.</li>
            <li>Call internal APIs for live offers, tracked results, and admin refresh status.</li>
            <li>Keep browser logic separate from odds ingestion, tracking files, and model workflows.</li>
          </ul>
        </div>
        <div className="info-list">
          <h3>Next steps after shell</h3>
          <ul className="simple-list">
            <li>Connect `/mlb` to yesterday tracked pitcher results.</li>
            <li>Add sport and event API contracts in Spring Boot.</li>
            <li>Create authenticated admin status and refresh pages.</li>
          </ul>
        </div>
      </section>
    </>
  );
}
