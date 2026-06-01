import { Metadata } from "next";

import { SportLandingPage } from "../../components/sport-landing-page";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "NBA"
};

const content = sportPageContent.nba;

export default function NbaPage() {
  return (
    <SportLandingPage
      content={content}
      sideTitle="NBA shell focus"
      sideDescription="The NBA shell is built for rapidly changing same-day markets, where lineup status and sportsbook coverage drive the user experience."
      stats={[
        { label: "Primary UX axis", value: "Slate first" },
        { label: "Key dependency", value: "Lineup freshness" },
        { label: "API shape", value: "Events | Players | Offers" }
      ]}
    />
  );
}
