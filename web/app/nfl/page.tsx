import { Metadata } from "next";

import { SportLandingPage } from "../../components/sport-landing-page";
import { sportPageContent } from "../../lib/sport-page-content";

export const metadata: Metadata = {
  title: "NFL"
};

const content = sportPageContent.nfl;

export default function NflPage() {
  return (
    <SportLandingPage
      content={content}
      sideTitle="NFL shell focus"
      sideDescription="NFL needs a weekly navigation pattern rather than a purely daily one, but it should still fit the same public shell and backend contract model."
      stats={[
        { label: "Primary cadence", value: "Weekly slate" },
        { label: "Key pages", value: "Week | Event | Player prop" },
        { label: "Backend pattern", value: "Shared multi-sport API" }
      ]}
    />
  );
}
