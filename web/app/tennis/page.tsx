import { Metadata } from "next";

import { SportLandingPage } from "../../components/sport-landing-page";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "Tennis"
};

const content = sportPageContent.tennis;

export default function TennisPage() {
  return (
    <SportLandingPage
      content={content}
      sideTitle="Why tennis fans use it"
      sideDescription="Tennis is a player-first sport, so the page is built to make standout matches and prop angles feel fast to reach."
      stats={[
        { label: "Best for", value: "Match-first browsing" },
        { label: "Watch first", value: "Featured players" },
        { label: "Visit style", value: "Tournament follow-through" }
      ]}
    />
  );
}
