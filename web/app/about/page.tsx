import { Metadata } from "next";

import { ApiConnectivityCard } from "../../components/api-connectivity-card";
import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";

export const metadata: Metadata = {
  title: "About"
};

export default function AboutPage() {
  return (
    <>
      <PageHero
        eyebrow="About"
        title="A public website shell sitting in front of internal sports workflows."
        description="This route explains the stack and boundaries behind the website MVP: Next.js on the frontend, a Spring Boot API behind it, and controlled access to tracked results and admin refresh tooling."
        sideTitle="Chosen stack"
        sideDescription="The website shell follows the accepted architecture decision record and keeps the browser separate from raw provider and filesystem concerns."
        stats={[
          { label: "Frontend", value: "Next.js + TypeScript" },
          { label: "Backend", value: "Java 21 + Spring Boot" },
          { label: "Hosting preference", value: "AWS" }
        ]}
      />

      <section className="page-block section-grid">
        <ApiConnectivityCard />
        <SectionCard title="Boundaries">
          <p>
            The frontend owns routing, page composition, and SEO-friendly public pages. The backend owns normalization, provider integrations, tracked-results adapters, and admin operations.
          </p>
        </SectionCard>
        <SectionCard title="Initial website responsibilities">
          <p>
            Home page, sport landing pages, game slates, player prop pages, yesterday results, and admin placeholders should all be served through internal APIs rather than direct file or provider access.
          </p>
        </SectionCard>
        <SectionCard title="Why this shell exists">
          <p>
            It gives the repo a clean public-facing website surface without entangling the existing MLB workflow code with browser concerns.
          </p>
        </SectionCard>
      </section>

      <section className="page-block info-list">
        <h3>Next implementation steps</h3>
        <ul className="feature-list">
          <li>Wire `/mlb` to internal endpoints that expose `pitcher_k` and `pitcher_bb` results from tracking data.</li>
          <li>Expand the staging runtime check into sport-specific pages once real API payloads exist.</li>
          <li>Add admin auth and refresh endpoints before exposing any operational controls in the UI.</li>
        </ul>
      </section>
    </>
  );
}
