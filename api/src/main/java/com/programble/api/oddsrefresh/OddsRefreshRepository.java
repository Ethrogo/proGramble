package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Optional;
import java.util.Set;

public interface OddsRefreshRepository {

	Optional<EventMatch> findMlbEvent(OffsetDateTime commenceTime, String homeTeam, String awayTeam);

	long upsertSportsbook(SportsbookUpsert sportsbook);

	long upsertPitcherStrikeoutsMarket(long sportId);

	long upsertPlayer(PlayerUpsert player);

	long ensurePlayerEventParticipant(PlayerEventParticipantUpsert participant);

	void upsertOffer(OfferUpsert offer);

	int deleteMissingOffers(long sportsbookId, long eventId, long marketId, Set<String> retainedSourceOfferIds);

	record EventMatch(
			long eventId,
			long sportId,
			LocalDate eventDate
	) {
	}

	record SportsbookUpsert(
			String code,
			String slug,
			String displayName,
			String regionCode
	) {
	}

	record PlayerUpsert(
			long sportId,
			String externalRef,
			String displayName
	) {
	}

	record PlayerEventParticipantUpsert(
			long eventId,
			long playerId,
			String roleCode,
			boolean home,
			boolean away,
			int sortOrder
	) {
	}

	record OfferUpsert(
			long sportsbookId,
			long eventId,
			long marketId,
			long eventParticipantId,
			BigDecimal lineValue,
			Integer americanPrice,
			BigDecimal decimalPrice,
			String selectionLabel,
			String sideCode,
			String outcomeType,
			OffsetDateTime availableAt,
			boolean live,
			String sourceOfferId
	) {
	}
}
