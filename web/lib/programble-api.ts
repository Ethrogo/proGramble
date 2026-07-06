export const PITCHER_STRIKEOUTS_MARKET_KEY = "pitcher_strikeouts";

export type ProgrambleSport = {
  code: string;
  slug: string;
  name: string;
};

export type ProgrambleCompetition = {
  code: string;
  slug: string;
  name: string;
  competitionType: string;
};

export type ProgrambleVenue = {
  name: string | null;
  city: string | null;
  countryCode: string | null;
};

export type ProgrambleEventParticipant = {
  id: number;
  type: string;
  roleCode: string;
  displayName: string;
  shortName: string;
  seedValue: number | null;
  sortOrder: number | null;
  isHome: boolean | null;
  isAway: boolean | null;
};

export type ProgrambleEventSummary = {
  id: number;
  slug: string;
  eventType: string;
  status: string;
  seasonLabel: string | null;
  roundLabel: string | null;
  scheduledStart: string;
  startTimeConfirmed: boolean;
  competition: ProgrambleCompetition;
  venue: ProgrambleVenue;
  participants: ProgrambleEventParticipant[];
};

export type ProgrambleEventListResponse = {
  sport: ProgrambleSport;
  date: string;
  count: number;
  events: ProgrambleEventSummary[];
};

export type ProgrambleEventDetail = {
  id: number;
  sport: ProgrambleSport;
  competition: ProgrambleCompetition;
  slug: string;
  externalRef: string | null;
  eventType: string;
  status: string;
  seasonLabel: string | null;
  roundLabel: string | null;
  scheduledStart: string;
  startTimeConfirmed: boolean;
  venue: ProgrambleVenue;
  participants: ProgrambleEventParticipant[];
};

export type ProgrambleOfferParticipant = {
  eventParticipantId: number;
  teamId: number | null;
  playerId: number | null;
  type: string;
  roleCode: string;
  displayName: string;
  shortName: string;
  sortOrder: number | null;
  isHome: boolean | null;
  isAway: boolean | null;
};

export type ProgrambleSportsbook = {
  id: number;
  code: string;
  slug: string;
  displayName: string;
  regionCode: string | null;
};

export type ProgrambleMarket = {
  id: number;
  code: string;
  slug: string;
  displayName: string;
  marketScope: string;
  statType: string | null;
  periodType: string | null;
  allowsOverUnder: boolean;
  allowsBinaryOutcome: boolean;
};

export type ProgrambleOffer = {
  id: number;
  event: {
    id: number;
    slug: string;
    externalRef: string | null;
    eventType: string;
    status: string;
    scheduledStart: string;
    startTimeConfirmed: boolean;
    sport: ProgrambleSport;
    competition: ProgrambleCompetition;
    venue: ProgrambleVenue;
  };
  sportsbook: ProgrambleSportsbook;
  market: ProgrambleMarket;
  participant: ProgrambleOfferParticipant | null;
  lineValue: number | null;
  priceAmerican: number | null;
  priceDecimal: number | null;
  selectionLabel: string;
  sideCode: string | null;
  outcomeType: string | null;
  availableAt: string;
  isLive: boolean;
};

export type ProgrambleEventOffersResponse = {
  event: ProgrambleOffer["event"];
  filters: {
    playerId: number | null;
    sportsbook: string | null;
    marketType: string | null;
  };
  count: number;
  offers: ProgrambleOffer[];
};

function normalizeBaseUrl(value: string | undefined): string | null {
  const trimmed = value?.trim();
  if (!trimmed) {
    return null;
  }

  return trimmed.replace(/\/+$/, "");
}

export function getProgrambleApiBaseUrl(): string | null {
  return normalizeBaseUrl(
    process.env.PROGRAMBLE_API_BASE_URL ?? process.env.NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL
  );
}

export class ProgrambleApiConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProgrambleApiConfigurationError";
  }
}

export class ProgrambleApiNotFoundError extends Error {
  readonly status: number;

  constructor(message: string, status = 404) {
    super(message);
    this.name = "ProgrambleApiNotFoundError";
    this.status = status;
  }
}

export class ProgrambleApiRequestError extends Error {
  readonly status: number;
  readonly url: string;

  constructor(message: string, status: number, url: string) {
    super(message);
    this.name = "ProgrambleApiRequestError";
    this.status = status;
    this.url = url;
  }
}

function buildApiUrl(
  path: string,
  queryParams?: Record<string, string | number | undefined | null>
): string {
  const baseUrl = getProgrambleApiBaseUrl();
  if (!baseUrl) {
    throw new ProgrambleApiConfigurationError(
      "Missing PROGRAMBLE_API_BASE_URL or NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL."
    );
  }

  const url = new URL(path, `${baseUrl}/`);

  if (queryParams) {
    for (const [key, value] of Object.entries(queryParams)) {
      if (value === undefined || value === null || value === "") {
        continue;
      }

      url.searchParams.set(key, String(value));
    }
  }

  return url.toString();
}

async function requestProgrambleApi<T>(
  path: string,
  queryParams?: Record<string, string | number | undefined | null>
): Promise<T> {
  const url = buildApiUrl(path, queryParams);
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      accept: "application/json"
    }
  });

  if (response.status === 404) {
    throw new ProgrambleApiNotFoundError(`The requested resource was not found: ${url}`);
  }

  if (!response.ok) {
    throw new ProgrambleApiRequestError(
      `The Programble API returned HTTP ${response.status} for ${url}.`,
      response.status,
      url
    );
  }

  return (await response.json()) as T;
}

export function getSportEvents(sport: string, date: string) {
  return requestProgrambleApi<ProgrambleEventListResponse>(
    `/api/v1/sports/${encodeURIComponent(sport)}/events`,
    { date }
  );
}

export function getEventDetail(eventId: number) {
  return requestProgrambleApi<ProgrambleEventDetail>(`/api/v1/events/${eventId}`);
}

export function getEventOffers(
  eventId: number,
  options?: {
    playerId?: number;
    sportsbook?: string;
    marketType?: string;
  }
) {
  return requestProgrambleApi<ProgrambleEventOffersResponse>(`/api/v1/events/${eventId}/offers`, {
    playerId: options?.playerId,
    sportsbook: options?.sportsbook,
    marketType: options?.marketType
  });
}
