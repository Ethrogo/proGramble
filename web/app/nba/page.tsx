import { Metadata } from "next";

import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "NBA"
};

const content = sportPageContent.nba;

export default function NbaPage() {
  return (
    <>
      <PageHero
        eyebrow={content.sport}
        title={content.title}
        description={content.description}
        sideTitle="NBA shell focus"
        sideDescription="The NBA shell is built for rapidly changing same-day markets, where lineup status and sportsbook coverage drive the user experience."
        stats={[
          { label: "Primary UX axis", value: "Slate first" },
          { label: "Key dependency", value: "Lineup freshness" },
          { label: "API shape", value: "Events | Players | Offers" }
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
