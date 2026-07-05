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
      sideTitle="Why NFL fans use it"
      sideDescription="Football rewards planning, so the page is built to make the week's biggest games and player props easy to revisit."
      stats={[
        { label: "Best for", value: "Weekend boards" },
        { label: "Watch first", value: "Big matchups" },
        { label: "Visit style", value: "Weekly planning" }
      ]}
    />
  );
}
