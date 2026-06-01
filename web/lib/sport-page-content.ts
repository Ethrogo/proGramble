export type SportPageContent = {
  slug: string;
  sport: string;
  title: string;
  description: string;
  marketFocus: string[];
  searchSummary: string;
  searchExamples: string[];
  recentEvents: Array<{ title: string; detail: string; status: string }>;
  featuredProps: Array<{ title: string; detail: string; signal: string }>;
  rankingsPlaceholder: Array<{ title: string; detail: string }>;
  trendingPlaceholder: Array<{ title: string; detail: string }>;
  modules: Array<{ title: string; body: string }>;
  adminNotes: string[];
};

export const sportPageContent: Record<string, SportPageContent> = {
  mlb: {
    slug: "mlb",
    sport: "MLB",
    title: "Pitcher props, game slates, and tracked baseball results.",
    description:
      "The MLB entry point is designed around daily starting pitchers, slate-level odds comparison, player prop pages, and yesterday's tracked outcomes for pitcher strikeouts and pitcher walks.",
    marketFocus: ["Pitcher strikeouts", "Pitcher walks", "Starter slate", "Yesterday's tracked results"],
    searchSummary:
      "Future MLB search should let users jump directly to pitchers, teams, and today's most relevant starter-driven props.",
    searchExamples: ["Search all sports", "Find a pitcher", "Jump to a team slate"],
    recentEvents: [
      {
        title: "Today's starter slate",
        detail: "Probable starters, first-pitch windows, and book coverage for the active board.",
        status: "Live slate placeholder"
      },
      {
        title: "Yesterday's tracked pitcher results",
        detail: "Bridge `pitcher_k` and `pitcher_bb` outcomes from internal tracking into the public results flow.",
        status: "API-backed target"
      },
      {
        title: "Featured series board",
        detail: "Cluster same-day games by matchup so users can move from slate to event to pitcher prop cleanly.",
        status: "Design stub"
      }
    ],
    featuredProps: [
      {
        title: "Pitcher strikeouts",
        detail: "Lead market for model-backed MLB discovery and event-level entry points.",
        signal: "Primary launch market"
      },
      {
        title: "Pitcher walks",
        detail: "Supports the `pitcher_bb` workflow and yesterday-results storytelling.",
        signal: "Tracked market"
      },
      {
        title: "Starter outs recorded",
        detail: "Useful extension once the game slate and pitcher pages are stable.",
        signal: "Next expansion"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Top projected strikeout spots",
        detail: "Future ranking block for the strongest same-day pitcher strikeout edges."
      },
      {
        title: "Most active books",
        detail: "Placeholder for where MLB prop coverage is deepest on a given slate."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending pitchers",
        detail: "Will highlight the most-viewed pitchers and props on the board."
      },
      {
        title: "Market movement watch",
        detail: "Reserved for later line-movement and freshness indicators."
      }
    ],
    modules: [
      {
        title: "Game slate page",
        body: "Show today's MLB slate with probable starters, books posting live lines, and links into event and prop detail pages."
      },
      {
        title: "Player prop page",
        body: "Display market offers, model context, and later historical movement for a single pitcher prop market."
      },
      {
        title: "Tracked results bridge",
        body: "Expose yesterday's pitcher_k and pitcher_bb outcomes through the Spring Boot API instead of reading tracking files from the browser."
      }
    ],
    adminNotes: [
      "Refresh starter slate and live odds",
      "Publish yesterday tracked MLB results",
      "Inspect API quota and fallback diagnostics"
    ]
  },
  nba: {
    slug: "nba",
    sport: "NBA",
    title: "Fast-moving player markets with lineup-sensitive slate views.",
    description:
      "The NBA shell is structured for same-day game slates, player prop discovery, and eventual injury-aware projections served through a stable internal API.",
    marketFocus: ["Points", "Rebounds", "Assists", "Combo props"],
    searchSummary:
      "NBA search will need both a site-wide entry point and a sport-specific jump path for teams, players, and same-day games.",
    searchExamples: ["Search all sports", "Find a player", "Open today's slate"],
    recentEvents: [
      {
        title: "Tonight's slate window",
        detail: "Game cards ordered around start times, lineup certainty, and prop availability.",
        status: "Live slate placeholder"
      },
      {
        title: "Late injury swing board",
        detail: "Reserve a section for games where status changes materially affect prop relevance.",
        status: "Design stub"
      },
      {
        title: "Spotlight matchup",
        detail: "Single event module for marquee games with concentrated player prop demand.",
        status: "Featured event placeholder"
      }
    ],
    featuredProps: [
      {
        title: "Points",
        detail: "Primary player-market entry path for star-driven game discovery.",
        signal: "Core market"
      },
      {
        title: "Rebounds + assists",
        detail: "Combo markets should group cleanly under the same player page framework.",
        signal: "High-interest combo"
      },
      {
        title: "Three-pointers made",
        detail: "Natural candidate for featured-market rotation on busy slates.",
        signal: "Expansion market"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Top projection risers",
        detail: "Placeholder for players moving up due to lineup or usage changes."
      },
      {
        title: "Most active games",
        detail: "Future ranking for games drawing the deepest market coverage."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending players",
        detail: "Most-viewed player pages and props on the current slate."
      },
      {
        title: "News-sensitive spots",
        detail: "Placeholder for highlighting markets affected by late injury reports."
      }
    ],
    modules: [
      {
        title: "Slate-first browsing",
        body: "Prioritize today's games, tip times, and featured player markets before deeper player and event detail pages."
      },
      {
        title: "Player market clusters",
        body: "Group props by player and by market family so a single page can support points, rebounds, assists, and combinations."
      },
      {
        title: "Admin placeholders",
        body: "Reserve room for lineup refresh status, odds snapshots, and manual refresh controls."
      }
    ],
    adminNotes: [
      "Track lineup refresh lag",
      "Flag books with missing player markets",
      "Surface game-level data staleness"
    ]
  },
  nfl: {
    slug: "nfl",
    sport: "NFL",
    title: "Weekly game boards, player props, and clean event navigation.",
    description:
      "NFL routes should support a wider weekly planning cadence than MLB or NBA while keeping the same website shell and API boundaries.",
    marketFocus: ["Passing yards", "Receiving yards", "Rushing yards", "Touchdowns"],
    searchSummary:
      "NFL search should prioritize weekly navigation, letting users jump to teams, players, and the current week board without browsing every matchup.",
    searchExamples: ["Search all sports", "Find a team", "Open Week 1 board"],
    recentEvents: [
      {
        title: "Current week slate",
        detail: "Group games by week first, then expose featured matchups and prop-heavy spots.",
        status: "Weekly slate placeholder"
      },
      {
        title: "Prime-time spotlight",
        detail: "Dedicated slot for the highest-interest standalone game on the board.",
        status: "Featured event placeholder"
      },
      {
        title: "Injury-report watchlist",
        detail: "Future event rail for games where injuries materially affect prop value.",
        status: "Design stub"
      }
    ],
    featuredProps: [
      {
        title: "Passing yards",
        detail: "Anchor market for quarterback discovery and weekly slate sorting.",
        signal: "Core market"
      },
      {
        title: "Receiving yards",
        detail: "High-traffic player page candidate for featured-matchup browsing.",
        signal: "Primary skill-position market"
      },
      {
        title: "Anytime touchdown",
        detail: "Good candidate for trending modules and market-group experimentation.",
        signal: "Engagement driver"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Top weekly looks",
        detail: "Reserved for best-rated games or props once ranking logic is live."
      },
      {
        title: "Most covered books",
        detail: "Future view of where weekly prop coverage is broadest."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending skill players",
        detail: "Most-viewed props and player pages for the current week."
      },
      {
        title: "Line movement watch",
        detail: "Placeholder for late-week market changes worth surfacing."
      }
    ],
    modules: [
      {
        title: "Weekly slate structure",
        body: "Support browsing by week, then by game, then by player market with stable URL patterns."
      },
      {
        title: "Prop pages with context",
        body: "Leave room for matchup context, book comparison, and later tracked outcomes."
      },
      {
        title: "Controlled expansion",
        body: "Reuse the same internal API contract shapes rather than special-casing the frontend for every sport."
      }
    ],
    adminNotes: [
      "Publish weekly slate status",
      "Track inactive-report updates",
      "Record missing market coverage by event"
    ]
  },
  tennis: {
    slug: "tennis",
    sport: "Tennis",
    title: "Match-centric browsing for ATP and WTA player markets.",
    description:
      "Tennis is modeled as a shared shell for ATP and WTA with match pages, player pages, and tour-aware sport landing content.",
    marketFocus: ["Aces", "Double faults", "Match winner", "Games won"],
    searchSummary:
      "Tennis search should help users jump by player, tournament, and tour while keeping ATP and WTA under one top-level shell.",
    searchExamples: ["Search all sports", "Find a player", "Filter ATP or WTA"],
    recentEvents: [
      {
        title: "Today's match slate",
        detail: "Upcoming matches grouped by tournament, tour, and start window.",
        status: "Live slate placeholder"
      },
      {
        title: "Featured court matches",
        detail: "Highlight the most active ATP or WTA matches for quick entry into player markets.",
        status: "Featured event placeholder"
      },
      {
        title: "Tournament progress board",
        detail: "Future section for round progression and carryover interest.",
        status: "Design stub"
      }
    ],
    featuredProps: [
      {
        title: "Aces",
        detail: "Player-centric prop category that maps cleanly to individual match pages.",
        signal: "Core prop"
      },
      {
        title: "Double faults",
        detail: "Useful secondary market for surfacing match-specific player tendencies.",
        signal: "Secondary prop"
      },
      {
        title: "Games won",
        detail: "Good bridge between match outcome and player-level market browsing.",
        signal: "Expansion market"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Top players to watch",
        detail: "Placeholder for ranking the most interesting ATP and WTA spots."
      },
      {
        title: "Tournament depth",
        detail: "Future comparison of where market coverage is strongest."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending matches",
        detail: "Most-viewed ATP and WTA event pages on the current slate."
      },
      {
        title: "Player watchlist",
        detail: "Reserved for frequently searched players and matchups."
      }
    ],
    modules: [
      {
        title: "Tour split",
        body: "Use the tennis landing page as the shared top-level entry, then branch to ATP and WTA filters through the API."
      },
      {
        title: "Match-first event model",
        body: "The shell assumes event detail pages are match-centric instead of team-centric."
      },
      {
        title: "Unified player identity",
        body: "Keep the backend responsible for player identity and normalization across tours."
      }
    ],
    adminNotes: [
      "Refresh tournament slate",
      "Flag suspended or delayed matches",
      "Track ATP vs WTA market coverage separately"
    ]
  },
  golf: {
    slug: "golf",
    sport: "Golf",
    title: "Tournament and round views for PGA and LPGA markets.",
    description:
      "Golf uses the same shell but needs tournament and round framing instead of single-game slates. The route is intentionally broad enough for PGA and LPGA expansion.",
    marketFocus: ["Round score", "Birdies", "Matchups", "Placement markets"],
    searchSummary:
      "Golf search should let users move directly to a golfer, tournament, or round market while still working under one PGA/LPGA top-level route.",
    searchExamples: ["Search all sports", "Find a golfer", "Open the current tournament"],
    recentEvents: [
      {
        title: "Tournament overview",
        detail: "Shared entry point for the active PGA or LPGA event with round-aware navigation.",
        status: "Tournament placeholder"
      },
      {
        title: "Round board",
        detail: "Surface the current round and featured golfer matchups without forcing users through deep navigation.",
        status: "Round module stub"
      },
      {
        title: "Leaderboard spotlight",
        detail: "Future section for players near the top of the board once live event data is wired.",
        status: "Design stub"
      }
    ],
    featuredProps: [
      {
        title: "Round score",
        detail: "Baseline golfer-market entry point for round-based browsing.",
        signal: "Core market"
      },
      {
        title: "Birdies",
        detail: "A natural featured prop for highlighting aggressive scoring spots.",
        signal: "Player prop"
      },
      {
        title: "Head-to-head matchups",
        detail: "Useful for balancing golfer pages with event-level tournament views.",
        signal: "Comparison market"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Top round positions",
        detail: "Reserved for strongest same-round golfer opportunities."
      },
      {
        title: "Most active tournaments",
        detail: "Future ranking of events with the broadest market coverage."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending golfers",
        detail: "Most-viewed golfer pages and matchup markets."
      },
      {
        title: "Movement watch",
        detail: "Placeholder for notable changes across round and placement markets."
      }
    ],
    modules: [
      {
        title: "Tournament overview",
        body: "Surface event-level context first, then expose round and player pages underneath."
      },
      {
        title: "Round-specific props",
        body: "Leave room for round-by-round availability and later leaderboard-aware presentation."
      },
      {
        title: "Future split support",
        body: "Keep PGA and LPGA as backend filters rather than separate top-level shells at the MVP stage."
      }
    ],
    adminNotes: [
      "Publish tournament feed status",
      "Track round refresh timing",
      "Surface incomplete book coverage"
    ]
  }
};
