import { Metadata } from "next";

import { SportLandingPage } from "../../components/sport-landing-page";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "MLB"
};

const content = sportPageContent.mlb;

export default function MlbPage() {
  return (
    <SportLandingPage
      content={content}
      sideTitle="Why MLB fans start here"
      sideDescription="Baseball moves every day, so the page is built to get users into featured starters, strong matchups, and pitcher props fast."
      stats={[
        { label: "Best for", value: "Daily pitcher props" },
        { label: "Watch first", value: "Starter spots" },
        { label: "Visit style", value: "Quick game-day scan" }
      ]}
    />
  );
}
