import { Metadata } from "next";

import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "Tennis"
};

const content = sportPageContent.tennis;

export default function TennisPage() {
  return (
    <>
      <PageHero
        eyebrow={content.sport}
        title={content.title}
        description={content.description}
        sideTitle="Shared ATP/WTA shell"
        sideDescription="The top-level tennis route stays broad so the backend can expose ATP and WTA distinctions without fragmenting the public shell too early."
        stats={[
          { label: "Tours", value: "ATP + WTA" },
          { label: "Primary unit", value: "Match event" },
          { label: "Identity model", value: "Player-centric" }
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
