import { Metadata } from "next";

import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "Golf"
};

const content = sportPageContent.golf;

export default function GolfPage() {
  return (
    <>
      <PageHero
        eyebrow={content.sport}
        title={content.title}
        description={content.description}
        sideTitle="PGA and LPGA shell"
        sideDescription="The golf route supports tournament-centric browsing while keeping PGA and LPGA as filters and API dimensions rather than separate shells in the MVP."
        stats={[
          { label: "Tours", value: "PGA + LPGA" },
          { label: "Primary unit", value: "Tournament / round" },
          { label: "Later extension", value: "Leaderboard-aware pages" }
        ]}
      />

      <section className="page-block">
        <div className="pill-row">
          {content.marketFocus.map((item) => (
            <span key={item} className="pill">
              {item}
            </span>
          ))}
        </div>
      </section>

      <section className="page-block section-grid">
        {content.modules.map((module) => (
          <SectionCard key={module.title} title={module.title}>
            <p>{module.body}</p>
          </SectionCard>
        ))}
      </section>

      <section className="page-block info-list">
        <h3>Operational notes</h3>
        <ul className="feature-list">
          {content.adminNotes.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </>
  );
}
