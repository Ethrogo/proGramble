package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.List;

public interface OddsApiClient {

	boolean isConfigured();

	List<MlbPitcherStrikeoutEvent> fetchMlbPitcherStrikeoutEvents();

	record MlbPitcherStrikeoutEvent(
			String sourceEventId,
			OffsetDateTime commenceTime,
			String homeTeam,
			String awayTeam,
			List<PitcherStrikeoutOffer> offers
	) {
	}

	record PitcherStrikeoutOffer(
			String sportsbookKey,
			String sportsbookDisplayName,
			OffsetDateTime availableAt,
			String pitcherName,
			String side,
			BigDecimal line,
			Integer americanPrice
	) {
	}
}
