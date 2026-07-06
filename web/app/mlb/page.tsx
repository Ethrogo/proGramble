import { Metadata } from "next";

import { MlbSlatePage } from "../../components/mlb-slate-page";
import {
  addDays,
  getEasternDate,
  getTodayInEasternTime,
  isIsoDateString
} from "../../lib/event-display";
import {
  getEventOffers,
  getSportEvents,
  PITCHER_STRIKEOUTS_MARKET_KEY,
  ProgrambleApiConfigurationError,
  ProgrambleApiRequestError,
  type ProgrambleEventListResponse,
  type ProgrambleEventSummary,
  type ProgrambleOffer
} from "../../lib/programble-api";

export const metadata: Metadata = {
  title: "MLB Slate"
};

export const dynamic = "force-dynamic";
export const revalidate = 0;

type PageSearchParams = {
  date?: string | string[];
};

function normalizeDateParam(value: string | string[] | undefined) {
  const candidate = Array.isArray(value) ? value[0] : value;
  return isIsoDateString(candidate) ? candidate : getTodayInEasternTime();
}

function mergeDailySlate(
  primarySlate: ProgrambleEventListResponse,
  rolloverSlate: ProgrambleEventListResponse,
  selectedDate: string
) {
  const eventMap = new Map<number, ProgrambleEventSummary>();

  for (const event of [...primarySlate.events, ...rolloverSlate.events]) {
    if (getEasternDate(event.scheduledStart) !== selectedDate) {
      continue;
    }

    eventMap.set(event.id, event);
  }

  const events = [...eventMap.values()].sort((left, right) =>
    left.scheduledStart.localeCompare(right.scheduledStart)
  );

  return {
    ...primarySlate,
    date: selectedDate,
    count: events.length,
    events
  };
}

function summarizeFeaturedBoard(event: ProgrambleEventSummary, offers: ProgrambleOffer[]) {
  const pitchers = new Map<string, string>();
  const books = new Set<string>();
  let latestOfferAt: string | null = null;
  let liveOfferCount = 0;

  for (const offer of offers) {
    if (offer.participant?.displayName) {
      const pitcherKey = offer.participant.playerId?.toString() ?? offer.participant.displayName;
      pitchers.set(pitcherKey, offer.participant.displayName);
    }

    books.add(offer.sportsbook.displayName);
    if (!latestOfferAt || offer.availableAt > latestOfferAt) {
      latestOfferAt = offer.availableAt;
    }

    if (offer.isLive) {
      liveOfferCount += 1;
    }
  }

  return {
    event,
    offerCount: offers.length,
    pitcherCount: pitchers.size,
    sportsbookNames: [...books.values()].sort((left, right) => left.localeCompare(right)),
    featuredPitchers: [...pitchers.values()].slice(0, 3),
    latestOfferAt,
    liveOfferCount
  };
}

async function loadFeaturedBoards(events: ProgrambleEventSummary[]) {
  const offerResponses = await Promise.all(
    events.slice(0, 6).map(async (event) => {
      try {
        const response = await getEventOffers(event.id, {
          marketType: PITCHER_STRIKEOUTS_MARKET_KEY
        });

        if (response.count === 0) {
          return null;
        }

        return summarizeFeaturedBoard(event, response.offers);
      } catch {
        return null;
      }
    })
  );

  return offerResponses.filter((value) => value !== null).slice(0, 3);
}

export default async function MlbPage({
  searchParams
}: {
  searchParams?: Promise<PageSearchParams>;
}) {
  const resolvedSearchParams = searchParams ? await searchParams : undefined;
  const selectedDate = normalizeDateParam(resolvedSearchParams?.date);

  try {
    const [primarySlate, rolloverSlate] = await Promise.all([
      getSportEvents("mlb", selectedDate),
      getSportEvents("mlb", addDays(selectedDate, 1))
    ]);

    const slate = mergeDailySlate(primarySlate, rolloverSlate, selectedDate);
    const featuredBoards = await loadFeaturedBoards(slate.events);

    return (
      <MlbSlatePage
        selectedDate={selectedDate}
        slate={slate}
        featuredBoards={featuredBoards}
      />
    );
  } catch (error) {
    const message =
      error instanceof ProgrambleApiConfigurationError || error instanceof ProgrambleApiRequestError
        ? error.message
        : "The slate could not be loaded from the API right now.";

    return (
      <MlbSlatePage
        selectedDate={selectedDate}
        slate={null}
        featuredBoards={[]}
        errorMessage={message}
      />
    );
  }
}
