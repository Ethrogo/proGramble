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
      sideTitle="Why golf fans use it"
      sideDescription="Golf asks for a calmer layout, so the page is shaped to make tournaments, rounds, and golfer angles easier to follow."
      stats={[
        { label: "Best for", value: "Tournament boards" },
        { label: "Watch first", value: "Round angles" },
        { label: "Visit style", value: "Weekend tracking" }
      ]}
    />
  );
}
