package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Optional;
import java.util.Set;

public interface OddsRefreshRepository {

	Optional<EventMatch> findMlbEvent(OffsetDateTime commenceTime, String homeTeam, String awayTeam);

	Optional<EventMatch> findMlbEventByTeamCodes(LocalDate eventDate, String teamCode, String opponentCode);

	long ensureMlbSport();

	long ensureMlbCompetition(long sportId);

	long upsertTeam(TeamUpsert team);

	void ensureCompetitionTeam(CompetitionTeamUpsert competitionTeam);

	EventMatch upsertMlbEvent(EventUpsert event);

	long ensureTeamEventParticipant(TeamEventParticipantUpsert participant);

	long upsertSportsbook(SportsbookUpsert sportsbook);

	long upsertPitcherStrikeoutsMarket(long sportId);

	long upsertPlayer(PlayerUpsert player);

	long ensurePlayerEventParticipant(PlayerEventParticipantUpsert participant);

	void upsertOffer(OfferUpsert offer);

	int deleteMissingOffers(long sportsbookId, long eventId, long marketId, Set<String> retainedSourceOfferIds);

	record EventMatch(
			long eventId,
			long sportId,
			LocalDate eventDate,
			OffsetDateTime scheduledStart,
			long homeTeamId,
			long awayTeamId,
			String homeTeamCode,
			String awayTeamCode
	) {
	}

	record TeamUpsert(
			long sportId,
			String externalRef,
			String code,
			String shortName,
			String fullName,
			String city,
			String countryCode
	) {
	}

	record CompetitionTeamUpsert(
			long competitionId,
			long teamId,
			String externalRef
	) {
	}

	record EventUpsert(
			long sportId,
			long competitionId,
			String externalRef,
			String slug,
			String eventType,
			String status,
			String seasonLabel,
			String roundLabel,
			OffsetDateTime scheduledStart,
			boolean startTimeConfirmed,
			String venueName,
			String venueCity,
			String venueCountryCode,
			long homeTeamId,
			long awayTeamId,
			String homeTeamCode,
			String awayTeamCode
	) {
	}

	record TeamEventParticipantUpsert(
			long eventId,
			long teamId,
			String roleCode,
			boolean home,
			boolean away,
			int sortOrder
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
