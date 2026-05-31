import { Metadata } from "next";

import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "MLB"
};

const content = sportPageContent.mlb;

export default function MlbPage() {
  return (
    <>
      <PageHero
        eyebrow={content.sport}
        title={content.title}
        description={content.description}
        sideTitle="Initial MLB website modules"
        sideDescription="The MLB route is the first production-facing sport shell because it already has tracked pitcher results and a daily workflow."
        stats={[
          { label: "Primary tracked markets", value: "pitcher_k / pitcher_bb" },
          { label: "Core routes next", value: "Slate • Event • Prop" },
          { label: "Backend source", value: "Spring Boot + tracking adapter" }
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

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>Admin placeholders</h3>
          <ul className="feature-list">
            {content.adminNotes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="info-list">
          <h3>API boundary</h3>
          <p>
            This page should eventually read from internal routes for today's slate, prop offers, and yesterday's tracked pitcher outcomes. It should not read `data/tracking` directly from the browser.
          </p>
        </div>
      </section>
    </>
  );
}
