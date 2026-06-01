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
      sideTitle="Initial MLB website modules"
      sideDescription="The MLB route is the first production-facing sport shell because it already has tracked pitcher results and a daily workflow."
      stats={[
        { label: "Primary tracked markets", value: "pitcher_k / pitcher_bb" },
        { label: "Core routes next", value: "Slate | Event | Prop" },
        { label: "Backend source", value: "Spring Boot + tracking adapter" }
      ]}
    />
  );
}
