import { Metadata } from "next";

import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "NFL"
};

const content = sportPageContent.nfl;

export default function NflPage() {
  return (
    <>
      <PageHero
        eyebrow={content.sport}
        title={content.title}
        description={content.description}
        sideTitle="NFL shell focus"
        sideDescription="NFL needs a weekly navigation pattern rather than a purely daily one, but it should still fit the same public shell and backend contract model."
        stats={[
          { label: "Primary cadence", value: "Weekly slate" },
          { label: "Key pages", value: "Week | Event | Player prop" },
          { label: "Backend pattern", value: "Shared multi-sport API" }
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
        <h3>Admin placeholders</h3>
        <ul className="feature-list">
          {content.adminNotes.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </>
  );
}
