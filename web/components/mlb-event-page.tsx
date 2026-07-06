import Link from "next/link";

import { PageHero } from "./page-hero";
import { SectionCard } from "./section-card";
import {
  formatLineValue,
  formatMatchupLabel,
  formatOdds,
  formatStartTime,
  formatUpdatedTime,
  formatVenueLabel,
  getTeamParticipants
} from "../lib/event-display";
import type {
  ProgrambleEventDetail,
  ProgrambleEventOffersResponse,
  ProgrambleOffer
} from "../lib/programble-api";

type GroupedBookOffer = {
  key: string;
  sportsbookName: string;
  over: ProgrambleOffer | null;
  under: ProgrambleOffer | null;
  otherSelections: ProgrambleOffer[];
};

type GroupedPitcherOffer = {
  key: string;
  pitcherName: string;
  lineValue: number | null;
  sortOrder: number;
  anyLive: boolean;
  latestOfferAt: string | null;
  books: GroupedBookOffer[];
};

function groupPitcherOffers(offers: ProgrambleOffer[]) {
  const groups = new Map<
    string,
    {
      key: string;
      pitcherName: string;
      lineValue: number | null;
      sortOrder: number;
      anyLive: boolean;
      latestOfferAt: string | null;
      books: Map<string, GroupedBookOffer>;
    }
  >();

  for (const offer of offers) {
    const participant = offer.participant;
    if (!participant) {
      continue;
    }

    const playerKey = participant.playerId ?? participant.displayName;
    const lineKey = offer.lineValue ?? "open";
    const groupKey = `${playerKey}:${lineKey}`;
    const existingGroup = groups.get(groupKey) ?? {
      key: groupKey,
      pitcherName: participant.displayName,
      lineValue: offer.lineValue,
      sortOrder: participant.sortOrder ?? 999,
      anyLive: false,
      latestOfferAt: null,
      books: new Map<string, GroupedBookOffer>()
    };

    const bookKey = offer.sportsbook.slug;
    const existingBook = existingGroup.books.get(bookKey) ?? {
      key: bookKey,
      sportsbookName: offer.sportsbook.displayName,
      over: null,
      under: null,
      otherSelections: []
    };

    if (offer.sideCode === "OVER") {
      existingBook.over = offer;
    } else if (offer.sideCode === "UNDER") {
      existingBook.under = offer;
    } else {
      existingBook.otherSelections.push(offer);
    }

    existingGroup.books.set(bookKey, existingBook);
    existingGroup.anyLive = existingGroup.anyLive || offer.isLive;
    if (!existingGroup.latestOfferAt || offer.availableAt > existingGroup.latestOfferAt) {
      existingGroup.latestOfferAt = offer.availableAt;
    }

    groups.set(groupKey, existingGroup);
  }

  return [...groups.values()]
    .map(
      (group): GroupedPitcherOffer => ({
        key: group.key,
        pitcherName: group.pitcherName,
        lineValue: group.lineValue,
        sortOrder: group.sortOrder,
        anyLive: group.anyLive,
        latestOfferAt: group.latestOfferAt,
        books: [...group.books.values()].sort((left, right) =>
          left.sportsbookName.localeCompare(right.sportsbookName)
        )
      })
    )
    .sort((left, right) => {
      if (left.sortOrder !== right.sortOrder) {
        return left.sortOrder - right.sortOrder;
      }

      if (left.pitcherName !== right.pitcherName) {
        return left.pitcherName.localeCompare(right.pitcherName);
      }

      return (left.lineValue ?? 0) - (right.lineValue ?? 0);
    });
}

function countSportsbooks(offers: ProgrambleOffer[]) {
  return new Set(offers.map((offer) => offer.sportsbook.slug)).size;
}

export function MlbEventPage({
  event,
  offers,
  offersErrorMessage
}: {
  event: ProgrambleEventDetail;
  offers: ProgrambleEventOffersResponse | null;
  offersErrorMessage?: string;
}) {
  const groupedOffers = groupPitcherOffers(offers?.offers ?? []);
  const teams = getTeamParticipants(event.participants);
  const matchup = formatMatchupLabel(event.participants);

  return (
    <>
      <PageHero
        eyebrow="Pitcher Strikeout Board"
        title={matchup}
        description="Compare posted pitcher strikeout prices for this matchup across the books currently feeding the live MLB refresh flow."
        actions={[
          { href: "/mlb", label: "Back to MLB slate", variant: "secondary" },
          { href: "#offers-board", label: "Jump to prop offers", variant: "primary" }
        ]}
        sideTitle="Game snapshot"
        sideDescription="This page is driven by the live events and offers APIs so the board reflects the current event catalog and pitcher prop feed."
        stats={[
          { label: "First pitch", value: formatStartTime(event.scheduledStart) },
          { label: "Pitchers on board", value: `${groupedOffers.length}` },
          { label: "Sportsbooks", value: `${countSportsbooks(offers?.offers ?? [])}` }
        ]}
      />

      <section className="page-block split-layout">
        <div className="info-list">
          <p className="eyebrow">Matchup details</p>
          <h3>{event.competition.name}</h3>
          <p>{event.roundLabel ?? event.eventType}</p>
          <div className="pill-row">
            <span className="pill">{event.status}</span>
            <span className="pill">{formatVenueLabel(event.venue)}</span>
          </div>
        </div>
        <div className="info-list">
          <p className="eyebrow">Teams</p>
          <h3>Who is on the card</h3>
          <div className="pill-row">
            {teams.map((team) => (
              <span key={team.id} className="pill">
                {team.displayName}
              </span>
            ))}
          </div>
          <p className="card-footnote">
            Open another game from the <Link href="/mlb">MLB slate</Link> if you want a different prop board.
          </p>
        </div>
      </section>

      {offersErrorMessage ? (
        <section className="page-block">
          <SectionCard title="Pitcher props are temporarily unavailable">
            <p>{offersErrorMessage}</p>
          </SectionCard>
        </section>
      ) : null}

      <section id="offers-board" className="page-block">
        <div className="section-heading">
          <p className="eyebrow">Player prop offers</p>
          <h2 className="section-title">Pitcher strikeout lines for this event.</h2>
        </div>
        {groupedOffers.length > 0 ? (
          <div className="offer-board">
            {groupedOffers.map((group) => (
              <article key={group.key} className="card offer-card">
                <div className="headline-row">
                  <div>
                    <h3>{group.pitcherName}</h3>
                    <p className="offer-line">{formatLineValue(group.lineValue)} strikeouts</p>
                  </div>
                  <div className="pill-row compact-pills">
                    {group.anyLive ? <span className="pill">Live</span> : null}
                    {group.latestOfferAt ? <span className="pill">Updated {formatUpdatedTime(group.latestOfferAt)}</span> : null}
                  </div>
                </div>
                <div className="book-grid">
                  {group.books.map((book) => (
                    <div key={book.key} className="book-card">
                      <strong>{book.sportsbookName}</strong>
                      <div className="book-price-row">
                        <span>Over</span>
                        <span>{formatOdds(book.over?.priceAmerican ?? null)}</span>
                      </div>
                      <div className="book-price-row">
                        <span>Under</span>
                        <span>{formatOdds(book.under?.priceAmerican ?? null)}</span>
                      </div>
                      {book.otherSelections.length > 0 ? (
                        <ul className="simple-list inline-list">
                          {book.otherSelections.map((offer) => (
                            <li key={offer.id}>
                              {offer.selectionLabel} {formatOdds(offer.priceAmerican)}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <SectionCard title="No strikeout offers posted yet">
            <p>
              This matchup is in the event catalog, but no pitcher strikeout prices are attached to it yet. Check back later as books publish lines.
            </p>
          </SectionCard>
        )}
      </section>
    </>
  );
}
