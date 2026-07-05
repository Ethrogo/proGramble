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
      sideTitle="Why NBA fans use it"
      sideDescription="Basketball nights move fast, so the page is designed to make top games and star-player props easier to spot."
      stats={[
        { label: "Best for", value: "Game-night props" },
        { label: "Watch first", value: "Star players" },
        { label: "Visit style", value: "Nightly check-in" }
      ]}
    />
  );
}
