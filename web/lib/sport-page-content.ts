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
    title: "Your faster way into today's MLB pitcher props and baseball storylines.",
    description:
      "Start with the starters, rivalry matchups, and pitcher markets baseball fans care about most without getting lost in a crowded board.",
    marketFocus: ["Pitcher strikeouts", "Pitcher walks", "Starter spotlights", "Game-night favorites"],
    searchSummary:
      "Jump straight from a favorite team or pitcher into the matchups and prop angles that matter most on the day's card.",
    searchExamples: ["Find a pitcher", "Jump to tonight's games", "Track your team"],
    recentEvents: [
      {
        title: "Today's starter spotlight",
        detail: "A fast way to scan the arms shaping the board before first pitch.",
        status: "Today"
      },
      {
        title: "Featured rivalry board",
        detail: "Bring the most interesting same-day matchup to the front so users know where to start.",
        status: "Popular"
      },
      {
        title: "Yesterday's baseball storylines",
        detail: "Keep recent results and standout performances close enough to spark the next visit.",
        status: "Coming soon"
      }
    ],
    featuredProps: [
      {
        title: "Pitcher strikeouts",
        detail: "The signature baseball prop for users who want a quick read on the day's best pitcher spots.",
        signal: "Fan favorite"
      },
      {
        title: "Pitcher walks",
        detail: "A sharper secondary angle for users who like pitching volatility and matchup nuance.",
        signal: "Smart add-on"
      },
      {
        title: "Starter outs recorded",
        detail: "A natural next stop for users who want more depth once the main pitcher page is open.",
        signal: "On deck"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Know where to start",
        detail: "A baseball page should quickly surface the pitchers and games most worth a first click."
      },
      {
        title: "Stay close to the daily rhythm",
        detail: "Baseball rewards repeat visits, so the page should feel useful every afternoon and night."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending pitchers",
        detail: "Highlight the starters and props drawing the most attention on a busy slate."
      },
      {
        title: "Watch list matchups",
        detail: "Keep the biggest series and same-day pitching spots easy to spot."
      }
    ],
    modules: [
      {
        title: "Daily slate first",
        body: "Lead with the games and starters people want to scan before they settle on a prop."
      },
      {
        title: "Pitcher pages that feel worth opening",
        body: "Make every featured pitcher page feel like a natural destination instead of another long list."
      },
      {
        title: "Recent baseball context",
        body: "Keep yesterday's takeaways and recent momentum close enough to support the next decision."
      }
    ],
    adminNotes: [
      "Featured pitcher pages worth revisiting every day",
      "Cleaner game cards built around top matchups",
      "More baseball prop depth as the experience expands"
    ]
  },
  nba: {
    slug: "nba",
    sport: "NBA",
    title: "A cleaner way to scan NBA player props and game-night boards.",
    description:
      "The NBA experience is built for people who want the big games, star props, and nightly storylines in front of them quickly.",
    marketFocus: ["Points", "Rebounds", "Assists", "Combo props"],
    searchSummary:
      "Go straight to a player, team, or game-night board without bouncing through a dozen pages first.",
    searchExamples: ["Find a player", "Open tonight's slate", "Track a matchup"],
    recentEvents: [
      {
        title: "Tonight's headline slate",
        detail: "Put the best games and most popular player markets at the front of the experience.",
        status: "Tonight"
      },
      {
        title: "Late-news watch list",
        detail: "Help users quickly spot where the board could change the feel of the night.",
        status: "Popular"
      },
      {
        title: "Spotlight matchup",
        detail: "Create an obvious entry point for the game everyone wants to check first.",
        status: "Coming soon"
      }
    ],
    featuredProps: [
      {
        title: "Points",
        detail: "The fastest path into the star-player props that drive most NBA browsing.",
        signal: "Core market"
      },
      {
        title: "Rebounds + assists",
        detail: "A natural second look for fans who want more than the headline scoring line.",
        signal: "High-interest combo"
      },
      {
        title: "Three-pointers made",
        detail: "A fun, high-attention market that keeps the page feeling active on busy nights.",
        signal: "On deck"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Built for quick reads",
        detail: "NBA pages should help users pick up the shape of the slate at a glance."
      },
      {
        title: "Strong for repeat use",
        detail: "The right mix of star power and nightly motion keeps people checking back."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending players",
        detail: "Show which players and props are pulling the most attention right now."
      },
      {
        title: "Games heating up",
        detail: "Keep the loudest game-night storylines close to the front."
      }
    ],
    modules: [
      {
        title: "Slate-first browsing",
        body: "Let the night's best games lead the page before users dive deeper into player props."
      },
      {
        title: "Star-player prop hubs",
        body: "Group the most relevant markets together so a player page feels focused instead of scattered."
      },
      {
        title: "Energy all night",
        body: "The page should feel alive enough to support quick pre-tip visits and second checks later on."
      }
    ],
    adminNotes: [
      "Nightly featured-player pages",
      "Clearer game cards for big national spots",
      "More prop variety as basketball coverage grows"
    ]
  },
  nfl: {
    slug: "nfl",
    sport: "NFL",
    title: "Weekly NFL boards built for marquee matchups and player props.",
    description:
      "The NFL experience should help fans move from the weekly board into the biggest games and most interesting props without wasting time.",
    marketFocus: ["Passing yards", "Receiving yards", "Rushing yards", "Touchdowns"],
    searchSummary:
      "Jump to a favorite team, a star player, or the week's most important game without scanning the full schedule first.",
    searchExamples: ["Find a team", "Open this week's board", "Spotlight a player"],
    recentEvents: [
      {
        title: "This week's board",
        detail: "Make the headline games and most bet props the first thing users see.",
        status: "This week"
      },
      {
        title: "Prime-time spotlight",
        detail: "Give the biggest standalone matchup a front-row spot on the page.",
        status: "Fan favorite"
      },
      {
        title: "Weekend watch list",
        detail: "Keep the players and games people are most likely to check again before kickoff.",
        status: "Coming soon"
      }
    ],
    featuredProps: [
      {
        title: "Passing yards",
        detail: "The cleanest way to lead users from the weekly board into quarterback-driven matchups.",
        signal: "Core market"
      },
      {
        title: "Receiving yards",
        detail: "A natural skill-position entry point when fans want star names first.",
        signal: "Primary skill-position market"
      },
      {
        title: "Anytime touchdown",
        detail: "A high-interest market that adds energy and broad appeal to weekly browsing.",
        signal: "Engagement driver"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Fits the weekly habit",
        detail: "NFL users often plan ahead, so the page should make the whole week feel approachable."
      },
      {
        title: "Built around the big games",
        detail: "A strong NFL page should always make the marquee spots feel easy to find."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending skill players",
        detail: "Surface the names and props getting the most weekly attention."
      },
      {
        title: "Weekend buzz",
        detail: "Keep the conversation around the biggest games close to the front."
      }
    ],
    modules: [
      {
        title: "Weekly board first",
        body: "Lead with the shape of the week, then pull users into the games and props worth extra attention."
      },
      {
        title: "Player pages with purpose",
        body: "Make prop pages feel like focused destinations for quarterbacks, stars, and breakout names."
      },
      {
        title: "Built for the weekend cycle",
        body: "Support early-week planning, midweek check-ins, and final pregame visits without losing clarity."
      }
    ],
    adminNotes: [
      "Big-game pages that feel worth sharing",
      "Cleaner player-prop discovery for weekends",
      "More weekly depth as football coverage expands"
    ]
  },
  tennis: {
    slug: "tennis",
    sport: "Tennis",
    title: "Match-first tennis browsing for ATP and WTA fans.",
    description:
      "The tennis experience is meant to feel quick, player-led, and easy to browse whether the attention is on ATP, WTA, or a single headline match.",
    marketFocus: ["Aces", "Double faults", "Match winner", "Games won"],
    searchSummary:
      "Move from a player or tournament into the matchups and prop angles worth watching without overcomplicating the path.",
    searchExamples: ["Find a player", "Browse today's matches", "Filter ATP or WTA"],
    recentEvents: [
      {
        title: "Today's match slate",
        detail: "Give users a clean starting point for the day's most interesting tennis matches.",
        status: "Today"
      },
      {
        title: "Featured court matches",
        detail: "Pull the biggest ATP and WTA matches to the front for quick access.",
        status: "Popular"
      },
      {
        title: "Tournament momentum",
        detail: "Keep standout runs and ongoing tournament storylines easy to revisit.",
        status: "Coming soon"
      }
    ],
    featuredProps: [
      {
        title: "Aces",
        detail: "A natural prop-first entry point for fans who follow individual player tendencies.",
        signal: "Core prop"
      },
      {
        title: "Double faults",
        detail: "A strong secondary angle that makes tennis pages feel richer without feeling crowded.",
        signal: "Secondary prop"
      },
      {
        title: "Games won",
        detail: "A smart bridge between overall match interest and player-focused browsing.",
        signal: "Expansion market"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Player-led by design",
        detail: "Tennis users often think in players first, so the experience should reflect that naturally."
      },
      {
        title: "Built for repeat checks",
        detail: "Tournaments evolve quickly, and the page should reward people who come back through the round."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending matches",
        detail: "Spotlight the ATP and WTA matches pulling the most interest."
      },
      {
        title: "Player watchlist",
        detail: "Keep the names and matchups people want most near the top."
      }
    ],
    modules: [
      {
        title: "One tennis home",
        body: "Let ATP and WTA live under one clean entry point so users stay oriented while browsing."
      },
      {
        title: "Match-first flow",
        body: "Build around the live appeal of single matches and the players driving them."
      },
      {
        title: "Tournament feel",
        body: "The page should make round progression and recurring match interest easy to follow."
      }
    ],
    adminNotes: [
      "Featured-player pages for ATP and WTA fans",
      "Cleaner tournament browsing across rounds",
      "More tennis prop depth as coverage expands"
    ]
  },
  golf: {
    slug: "golf",
    sport: "Golf",
    title: "Tournament and round pages built for golf fans who want a cleaner board.",
    description:
      "Golf should feel calm and organized, giving users a better way to move through tournaments, rounds, golfer matchups, and prop angles.",
    marketFocus: ["Round score", "Birdies", "Matchups", "Placement markets"],
    searchSummary:
      "Move straight to a golfer, current tournament, or round board without feeling buried in a long event menu.",
    searchExamples: ["Find a golfer", "Open the tournament", "Jump to round props"],
    recentEvents: [
      {
        title: "Tournament overview",
        detail: "A calmer front door into the active event, the top names, and the best places to start.",
        status: "This event"
      },
      {
        title: "Round board",
        detail: "Keep the current round and featured golfer angles close enough for quick check-ins.",
        status: "Popular"
      },
      {
        title: "Leaderboard spotlight",
        detail: "Highlight the golfers and storylines people will want to follow through the weekend.",
        status: "Coming soon"
      }
    ],
    featuredProps: [
      {
        title: "Round score",
        detail: "A clean starting point for fans who want round-by-round golfer focus.",
        signal: "Core market"
      },
      {
        title: "Birdies",
        detail: "A more personality-driven prop angle that adds life to tournament browsing.",
        signal: "Player prop"
      },
      {
        title: "Head-to-head matchups",
        detail: "A familiar comparison market that helps users move naturally between golfers and the event page.",
        signal: "Comparison market"
      }
    ],
    rankingsPlaceholder: [
      {
        title: "Built for calmer browsing",
        detail: "Golf pages should feel organized enough to support longer sessions without becoming noisy."
      },
      {
        title: "Good for weekend follow-through",
        detail: "The right tournament page gives users a reason to check back round after round."
      }
    ],
    trendingPlaceholder: [
      {
        title: "Trending golfers",
        detail: "Keep the golfers and matchups pulling the most attention easy to find."
      },
      {
        title: "Weekend watch",
        detail: "Surface the moments that make a tournament feel alive across several days."
      }
    ],
    modules: [
      {
        title: "Tournament first",
        body: "Let the main event page set the stage before users choose golfer-specific paths."
      },
      {
        title: "Round-specific flow",
        body: "Keep round props and golfer angles easy to revisit as the event progresses."
      },
      {
        title: "Built to stretch across tours",
        body: "Support PGA and LPGA fans with one clear experience that does not feel fragmented."
      }
    ],
    adminNotes: [
      "Cleaner tournament pages for repeat weekend visits",
      "Featured golfer pages and matchup paths",
      "More round-based depth as golf coverage grows"
    ]
  }
};
