import { Metadata } from "next";

import { ApiConnectivityCard } from "../../components/api-connectivity-card";
import { PageHero } from "../../components/page-hero";
import { SectionCard } from "../../components/section-card";

export const metadata: Metadata = {
  title: "Why ProGramble"
};

export default function AboutPage() {
  return (
    <>
      <PageHero
        eyebrow="Why ProGramble"
        title="A sports site built to win attention by being easier to use."
        description="ProGramble is being shaped as a cleaner way to browse featured slates, popular props, and the sports moments that matter most. The goal is simple: help users get to the right board faster and make the site worth revisiting every day."
        sideTitle="What the brand promises"
        sideDescription="Faster discovery, clearer presentation, and a sports-first experience that puts user momentum ahead of noise."
        stats={[
          { label: "Focus", value: "Featured slates" },
          { label: "Audience", value: "Prop-first fans" },
          { label: "Goal", value: "Come back tomorrow" }
        ]}
      />

      <section className="page-block section-grid">
        <SectionCard title="Built to save time">
          <p>
            The experience is designed to get users from broad browsing into the games, players, and props they care about with fewer clicks and less scanning.
          </p>
        </SectionCard>
        <SectionCard title="Focused on the plays people care about">
          <p>
            Instead of treating every market the same, ProGramble is organized around the sports storylines and prop categories that naturally drive attention.
          </p>
        </SectionCard>
        <SectionCard title="Made for daily habits">
          <p>
            The best sports sites reward repeat use. ProGramble is being shaped to feel familiar, quick, and easy to re-enter whenever the next slate drops.
          </p>
        </SectionCard>
        <SectionCard title="Growing across sports">
          <p>
            Baseball may be the first stop, but the experience is meant to scale naturally across basketball, football, tennis, and golf without feeling fragmented.
          </p>
        </SectionCard>
      </section>

      <section className="page-block split-layout">
        <div className="info-list">
          <h3>What users should expect</h3>
          <ul className="feature-list">
            <li>Sport landing pages that make the most relevant boards easy to find.</li>
            <li>Featured prop paths built around the categories fans already check first.</li>
            <li>A cleaner brand experience that feels useful even on a quick visit.</li>
          </ul>
        </div>
        <div className="info-list">
          <h3>Why that matters</h3>
          <ul className="simple-list">
            <li>Better first impressions turn curiosity into repeat usage.</li>
            <li>Cleaner browsing keeps attention on the sports instead of the interface.</li>
            <li>A strong public surface makes it easier to earn trust and grow an audience.</li>
          </ul>
        </div>
      </section>

      <section className="page-block">
        <ApiConnectivityCard />
      </section>
    </>
  );
}
