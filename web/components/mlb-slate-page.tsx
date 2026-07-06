import Link from "next/link";

import { PageHero } from "./page-hero";
import { SectionCard } from "./section-card";
import {
  addDays,
  formatMatchupLabel,
  formatShortStartTime,
  formatSlateDate,
  formatStartTime,
  formatUpdatedTime,
  formatVenueLabel,
  getTeamParticipants,
  getTodayInEasternTime
} from "../lib/event-display";
import type {
  ProgrambleEventListResponse,
  ProgrambleEventSummary
} from "../lib/programble-api";

type FeaturedBoard = {
  event: ProgrambleEventSummary;
  offerCount: number;
  pitcherCount: number;
  sportsbookNames: string[];
  featuredPitchers: string[];
  latestOfferAt: string | null;
  liveOfferCount: number;
};

export function MlbSlatePage({
  selectedDate,
  slate,
  featuredBoards,
  errorMessage
}: {
  selectedDate: string;
  slate: ProgrambleEventListResponse | null;
  featuredBoards: FeaturedBoard[];
  errorMessage?: string;
}) {
  const today = getTodayInEasternTime();
  const events = slate?.events ?? [];
  const totalFeaturedOffers = featuredBoards.reduce((sum, board) => sum + board.offerCount, 0);

  return (
    <>
      <PageHero
        eyebrow="MLB Slate"
        title="Today's MLB slate and live pitcher strikeout boards."
        description="Start with the games on deck, then jump straight into matchup pages showing live pitcher strikeout prices from the current refresh flow."
        actions={[
          { href: "#mlb-slate", label: "Browse today's games", variant: "primary" },
          { href: featuredBoards[0] ? `/mlb/events/${featuredBoards[0].event.id}` : "/about", label: featuredBoards[0] ? "Open a strikeout board" : "Why ProGramble", variant: "secondary" }
        ]}
        sideTitle="What is live right now"
        sideDescription="The baseball page now pulls real event slate data and live pitcher strikeout offers instead of static placeholders."
        stats={[
          { label: "Slate date", value: formatSlateDate(selectedDate) },
          { label: "Games on board", value: `${events.length}` },
          { label: "Strikeout prices", value: `${totalFeaturedOffers}` }
        ]}
      />

      <section className="page-block split-layout">
        <div className="info-list">
          <p className="eyebrow">Date navigation</p>
          <h3>Browse the baseball slate by day.</h3>
          <p>Times are shown in Eastern Time so the card reads like a real daily slate instead of a raw feed dump.</p>
          <div className="date-nav">
            <Link className="mini-link" href={`/mlb?date=${addDays(selectedDate, -1)}`}>
              Previous day
            </Link>
            {selectedDate !== today ? (
              <Link className="mini-link" href="/mlb">
                Jump to today
              </Link>
            ) : null}
            <Link className="mini-link" href={`/mlb?date=${addDays(selectedDate, 1)}`}>
              Next day
            </Link>
          </div>
        </div>
        <div className="info-list">
          <p className="eyebrow">What this page does</p>
          <h3>Find the game first, then compare the pitcher props.</h3>
          <ul className="simple-list">
            <li>Use the slate to move quickly into the matchup you care about.</li>
            <li>Open the event page to compare posted strikeout lines by sportsbook.</li>
            <li>Expect the board to stay focused on the live MLB pitcher workflow that already refreshes in the backend.</li>
          </ul>
        </div>
      </section>

      {errorMessage ? (
        <section className="page-block">
          <SectionCard title="Live MLB data is not available yet">
            <p>{errorMessage}</p>
            <p>
              Set <code>PROGRAMBLE_API_BASE_URL</code> or <code>NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL</code> for the web app so the staging site can reach the ECS API.
            </p>
          </SectionCard>
        </section>
      ) : null}

      {!errorMessage ? (
        <>
          <section className="page-block">
            <div className="section-heading">
              <p className="eyebrow">Featured strikeout boards</p>
              <h2 className="section-title">Real pitcher prop pages connected to the current refresh flow.</h2>
            </div>
            {featuredBoards.length > 0 ? (
              <div className="section-grid">
                {featuredBoards.map((board) => (
                  <SectionCard key={board.event.id} title={formatMatchupLabel(board.event.participants)}>
                    <p>
                      {board.offerCount} posted strikeout prices across {board.pitcherCount} pitcher
                      {board.pitcherCount === 1 ? "" : "s"} and {board.sportsbookNames.length} sportsbook
                      {board.sportsbookNames.length === 1 ? "" : "s"}.
                    </p>
                    <div className="pill-row">
                      {board.featuredPitchers.map((pitcher) => (
                        <span key={pitcher} className="pill">
                          {pitcher}
                        </span>
                      ))}
                      {board.liveOfferCount > 0 ? <span className="pill">Live now: {board.liveOfferCount}</span> : null}
                    </div>
                    <p className="card-footnote">
                      {board.latestOfferAt ? `Latest update ${formatUpdatedTime(board.latestOfferAt)}` : "Waiting for an updated price timestamp."}
                    </p>
                    <Link className="mini-link" href={`/mlb/events/${board.event.id}`}>
                      View strikeout offers
                    </Link>
                  </SectionCard>
                ))}
              </div>
            ) : (
              <SectionCard title="Strikeout prices are still populating">
                <p>
                  The slate is live, but the featured pitcher strikeout boards have not posted offers for this day yet. Once books publish lines, the event pages will surface them here.
                </p>
              </SectionCard>
            )}
          </section>

          <section id="mlb-slate" className="page-block">
            <div className="section-heading">
              <p className="eyebrow">Daily slate</p>
              <h2 className="section-title">Games available on {formatSlateDate(selectedDate)}.</h2>
            </div>
            {events.length > 0 ? (
              <div className="slate-grid">
                {events.map((event) => {
                  const teams = getTeamParticipants(event.participants);

                  return (
                    <article key={event.id} className="card slate-card">
                      <div className="headline-row">
                        <div>
                          <p className="eyebrow">MLB Event</p>
                          <h3 className="matchup-title">{formatMatchupLabel(event.participants)}</h3>
                        </div>
                        <span className="pill">{event.status}</span>
                      </div>
                      <p className="matchup-meta">
                        {event.roundLabel ?? event.competition.name}
                        {" · "}
                        {formatStartTime(event.scheduledStart)}
                      </p>
                      <p className="matchup-meta">{formatVenueLabel(event.venue)}</p>
                      {teams.length > 0 ? (
                        <div className="pill-row">
                          {teams.map((team) => (
                            <span key={team.id} className="pill">
                              {team.displayName}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      <div className="slate-card-actions">
                        <span className="pill">{formatShortStartTime(event.scheduledStart)}</span>
                        <Link className="mini-link" href={`/mlb/events/${event.id}`}>
                          Open matchup page
                        </Link>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <SectionCard title="No MLB games found for this date">
                <p>
                  There are no baseball events on the current slate for {formatSlateDate(selectedDate)}. Try the previous or next day to keep browsing.
                </p>
              </SectionCard>
            )}
          </section>
        </>
      ) : null}
    </>
  );
}
