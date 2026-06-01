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
      sideTitle="Shared ATP/WTA shell"
      sideDescription="The top-level tennis route stays broad so the backend can expose ATP and WTA distinctions without fragmenting the public shell too early."
      stats={[
        { label: "Tours", value: "ATP + WTA" },
        { label: "Primary unit", value: "Match event" },
        { label: "Identity model", value: "Player-centric" }
      ]}
    />
  );
}
