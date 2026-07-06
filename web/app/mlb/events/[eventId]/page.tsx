import { Metadata } from "next";
import { notFound } from "next/navigation";

import { MlbEventPage } from "../../../../components/mlb-event-page";
import { PageHero } from "../../../../components/page-hero";
import { SectionCard } from "../../../../components/section-card";
import {
  getEventDetail,
  getEventOffers,
  PITCHER_STRIKEOUTS_MARKET_KEY,
  ProgrambleApiConfigurationError,
  ProgrambleApiNotFoundError,
  ProgrambleApiRequestError
} from "../../../../lib/programble-api";

export const metadata: Metadata = {
  title: "MLB Matchup"
};

export const dynamic = "force-dynamic";
export const revalidate = 0;

type PageParams = {
  eventId: string;
};

function renderUnavailableState(message: string) {
  return (
    <>
      <PageHero
        eyebrow="MLB Matchup"
        title="Pitcher strikeout board unavailable"
        description="The matchup page is wired to the live event and offer APIs, but it could not load the current game data."
        sideTitle="What to check"
        sideDescription="This page depends on the staging API base URL and the MLB offers feed being reachable from the web app runtime."
        stats={[
          { label: "Status", value: "Unavailable" },
          { label: "Page type", value: "Event detail" },
          { label: "Feed", value: "Pitcher strikeouts" }
        ]}
      />
      <section className="page-block">
        <SectionCard title="Live data could not be loaded">
          <p>{message}</p>
        </SectionCard>
      </section>
    </>
  );
}

export default async function MlbEventDetailPage({
  params
}: {
  params: Promise<PageParams>;
}) {
  const resolvedParams = await params;
  const eventId = Number(resolvedParams.eventId);

  if (!Number.isFinite(eventId)) {
    notFound();
  }

  try {
    const event = await getEventDetail(eventId);
    if (event.sport.slug !== "mlb") {
      notFound();
    }

    try {
      const offers = await getEventOffers(eventId, {
        marketType: PITCHER_STRIKEOUTS_MARKET_KEY
      });

      return <MlbEventPage event={event} offers={offers} />;
    } catch (error) {
      if (
        error instanceof ProgrambleApiConfigurationError ||
        error instanceof ProgrambleApiRequestError
      ) {
        return (
          <MlbEventPage
            event={event}
            offers={null}
            offersErrorMessage={error.message}
          />
        );
      }

      throw error;
    }
  } catch (error) {
    if (error instanceof ProgrambleApiNotFoundError) {
      notFound();
    }

    if (
      error instanceof ProgrambleApiConfigurationError ||
      error instanceof ProgrambleApiRequestError
    ) {
      return renderUnavailableState(error.message);
    }

    throw error;
  }
}
