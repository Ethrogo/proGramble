import { Metadata } from "next";

import { SportLandingPage } from "../../components/sport-landing-page";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "Golf"
};

const content = sportPageContent.golf;

export default function GolfPage() {
  return (
    <SportLandingPage
      content={content}
      sideTitle="PGA and LPGA shell"
      sideDescription="The golf route supports tournament-centric browsing while keeping PGA and LPGA as filters and API dimensions rather than separate shells in the MVP."
      stats={[
        { label: "Tours", value: "PGA + LPGA" },
        { label: "Primary unit", value: "Tournament / round" },
        { label: "Later extension", value: "Leaderboard-aware pages" }
      ]}
    />
  );
}
