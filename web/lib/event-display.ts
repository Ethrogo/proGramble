import type {
  ProgrambleEventParticipant,
  ProgrambleVenue
} from "./programble-api";

export const EASTERN_TIME_ZONE = "America/New_York";

function formatParts(
  value: string | Date,
  options: Intl.DateTimeFormatOptions
) {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: EASTERN_TIME_ZONE,
    ...options
  }).format(value instanceof Date ? value : new Date(value));
}

function formatDateParts(date: Date) {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: EASTERN_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  });

  const parts = formatter.formatToParts(date);
  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;

  if (!year || !month || !day) {
    throw new Error("Unable to derive a calendar date.");
  }

  return `${year}-${month}-${day}`;
}

export function isIsoDateString(value: string | undefined): value is string {
  return Boolean(value && /^\d{4}-\d{2}-\d{2}$/.test(value));
}

export function getTodayInEasternTime() {
  return formatDateParts(new Date());
}

export function addDays(dateString: string, days: number) {
  const date = new Date(`${dateString}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

export function getEasternDate(value: string) {
  return formatDateParts(new Date(value));
}

export function formatSlateDate(dateString: string) {
  return formatParts(`${dateString}T12:00:00Z`, {
    weekday: "long",
    month: "long",
    day: "numeric"
  });
}

export function formatStartTime(value: string) {
  return formatParts(value, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short"
  });
}

export function formatShortStartTime(value: string) {
  return formatParts(value, {
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short"
  });
}

export function formatUpdatedTime(value: string) {
  return formatParts(value, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short"
  });
}

export function formatOdds(price: number | null) {
  if (price === null) {
    return "N/A";
  }

  return price > 0 ? `+${price}` : `${price}`;
}

export function formatLineValue(lineValue: number | null) {
  if (lineValue === null) {
    return "Open line";
  }

  return Number.isInteger(lineValue) ? `${lineValue}` : `${lineValue}`;
}

export function getTeamParticipants(participants: ProgrambleEventParticipant[]) {
  return participants
    .filter((participant) => participant.type === "TEAM")
    .sort((left, right) => (left.sortOrder ?? 999) - (right.sortOrder ?? 999));
}

export function formatMatchupLabel(participants: ProgrambleEventParticipant[]) {
  const teams = getTeamParticipants(participants);
  if (teams.length >= 2) {
    return `${teams[0].shortName} at ${teams[1].shortName}`;
  }

  if (participants.length >= 2) {
    return `${participants[0].shortName} vs ${participants[1].shortName}`;
  }

  return "Matchup to be announced";
}

export function formatVenueLabel(venue: ProgrambleVenue) {
  const values = [venue.name, venue.city].filter((value): value is string => Boolean(value));
  return values.length > 0 ? values.join(", ") : "Venue to be announced";
}
