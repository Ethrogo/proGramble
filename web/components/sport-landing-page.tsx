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
          { href: `/${content.slug}`, label: `Browse ${content.sport}`, variant: "primary" },
          { href: "/about", label: "Why ProGramble", variant: "secondary" }
        ]}
        sideTitle={sideTitle}
        sideDescription={sideDescription}
        stats={stats}
      />

      <section className="page-block split-layout">
        <div className="info-list">
          <p className="eyebrow">Quick start</p>
          <h3>Find the right board faster.</h3>
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
          <p className="eyebrow">Popular angles</p>
          <h3>What fans come here to track first</h3>
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
          <p className="eyebrow">Featured boards</p>
          <h2 className="section-title">Start with the matchups and moments drawing attention.</h2>
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
          <p className="eyebrow">Popular props</p>
          <h2 className="section-title">The prop categories people naturally want first.</h2>
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
          <p className="eyebrow">Reasons to return</p>
          <h3>Why this sport page should become part of a routine.</h3>
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
          <p className="eyebrow">What's catching on</p>
          <h3>Highlights that keep the page feeling active.</h3>
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

      <section className="page-block">
        <div className="section-heading">
          <p className="eyebrow">The experience</p>
          <h2 className="section-title">How ProGramble wants this sport to feel.</h2>
        </div>
        <div className="section-grid">
          {content.modules.map((module) => (
            <SectionCard key={module.title} title={module.title}>
              <p>{module.body}</p>
            </SectionCard>
          ))}
        </div>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>What users can look forward to</h3>
          <ul className="feature-list">
            {content.adminNotes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="info-list">
          <h3>Why ProGramble fits this sport</h3>
          <p>
            The goal is not to overwhelm people with everything at once. Each sport page is being shaped to spotlight the right boards, the right props, and the right reasons to come back tomorrow.
          </p>
        </div>
      </section>
    </>
  );
}
