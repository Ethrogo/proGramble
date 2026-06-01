import { PageHero } from "./page-hero";
import { SectionCard } from "./section-card";
import type { SportPageContent } from "../lib/sport-page-content";

export function SportLandingPage({
  content,
  sideTitle,
  sideDescription,
  stats
}: {
  content: SportPageContent;
  sideTitle: string;
  sideDescription: string;
  stats: Array<{ label: string; value: string }>;
}) {
  return (
    <>
      <PageHero
        eyebrow={content.sport}
        title={content.title}
        description={content.description}
        actions={[
          { href: `/${content.slug}`, label: `Open ${content.sport} board`, variant: "primary" },
          { href: "/about", label: "View product notes", variant: "secondary" }
        ]}
        sideTitle={sideTitle}
        sideDescription={sideDescription}
        stats={stats}
      />

      <section className="page-block split-layout">
        <div className="info-list">
          <p className="eyebrow">Future search</p>
          <h3>Global and {content.sport}-specific search should both live here.</h3>
          <p>{content.searchSummary}</p>
          <div className="pill-row">
            {content.searchExamples.map((example) => (
              <span key={example} className="pill">
                {example}
              </span>
            ))}
          </div>
        </div>
        <div className="info-list">
          <p className="eyebrow">Market focus</p>
          <h3>Featured market families</h3>
          <div className="pill-row">
            {content.marketFocus.map((item) => (
              <span key={item} className="pill">
                {item}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">Recent events</p>
          <h2 className="section-title">Reusable event modules for each sport landing page.</h2>
        </div>
        <div className="section-grid">
          {content.recentEvents.map((event) => (
            <SectionCard key={event.title} title={event.title}>
              <p>{event.detail}</p>
              <div className="pill-row">
                <span className="pill">{event.status}</span>
              </div>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">Featured props</p>
          <h2 className="section-title">Entry points for the prop pages that matter most.</h2>
        </div>
        <div className="section-grid">
          {content.featuredProps.map((prop) => (
            <SectionCard key={prop.title} title={prop.title}>
              <p>{prop.detail}</p>
              <div className="pill-row">
                <span className="pill">{prop.signal}</span>
              </div>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <p className="eyebrow">Rankings placeholder</p>
          <h3>Reserved for model-driven ranking modules.</h3>
          <div className="feature-stack">
            {content.rankingsPlaceholder.map((item) => (
              <div key={item.title} className="feature-row">
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="info-list">
          <p className="eyebrow">Trending placeholder</p>
          <h3>Reserved for high-interest markets and pages.</h3>
          <div className="feature-stack">
            {content.trendingPlaceholder.map((item) => (
              <div key={item.title} className="feature-row">
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.detail}</p>
                </div>
              </div>
            ))}
          </div>
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
          <h3>Operational placeholders</h3>
          <ul className="feature-list">
            {content.adminNotes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="info-list">
          <h3>Backend boundary</h3>
          <p>
            These landing pages are UI shells. Real recent events, featured props, rankings, and search results should come from the Spring Boot API rather than static browser-side provider calls.
          </p>
        </div>
      </section>
    </>
  );
}
