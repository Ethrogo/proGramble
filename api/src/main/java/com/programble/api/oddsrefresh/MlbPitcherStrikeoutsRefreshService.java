package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.text.Normalizer;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import com.programble.api.jobs.BackgroundJobResult;
import org.springframework.stereotype.Service;

@Service
public class MlbPitcherStrikeoutsRefreshService {

	private static final String MARKET_KEY = "pitcher_strikeouts";

	private final OddsApiClient oddsApiClient;
	private final MlbScheduleClient mlbScheduleClient;
	private final OddsRefreshRepository repository;
	private final MlbSportsDataRefreshService sportsDataRefreshService;

	public MlbPitcherStrikeoutsRefreshService(
			OddsApiClient oddsApiClient,
			MlbScheduleClient mlbScheduleClient,
			OddsRefreshRepository repository,
			MlbSportsDataRefreshService sportsDataRefreshService
	) {
		this.oddsApiClient = oddsApiClient;
		this.mlbScheduleClient = mlbScheduleClient;
		this.repository = repository;
		this.sportsDataRefreshService = sportsDataRefreshService;
	}

	public BackgroundJobResult refresh() {
		BackgroundJobResult baselineSportsData = this.sportsDataRefreshService.refresh();
		List<OddsApiClient.MlbPitcherStrikeoutEvent> oddsEvents = this.oddsApiClient.fetchMlbPitcherStrikeoutEvents();
		MlbSportsDataRefreshService.ScheduleRefreshSummary oddsEventScheduleRefresh =
				this.sportsDataRefreshService.ensureScheduleForOddsEvents(oddsEvents);
		Map<LocalDate, List<MlbScheduleClient.MlbScheduledGame>> schedulesByDate = new HashMap<>();

		int matchedEvents = 0;
		int unmatchedEvents = 0;
		int unmatchedPitchers = 0;
		int offersUpserted = 0;
		int offersRemoved = 0;
		Set<Long> sportsbookIdsTouched = new HashSet<>();
		Set<String> playerRefsTouched = new HashSet<>();
		Set<String> participantKeysTouched = new HashSet<>();
		long marketId = -1L;

		for (OddsApiClient.MlbPitcherStrikeoutEvent oddsEvent : oddsEvents) {
			Optional<OddsRefreshRepository.EventMatch> eventMatch = this.repository.findMlbEvent(
					oddsEvent.commenceTime(),
					oddsEvent.homeTeam(),
					oddsEvent.awayTeam()
			);

			if (eventMatch.isEmpty()) {
				unmatchedEvents++;
				continue;
			}

			List<MlbScheduleClient.MlbScheduledGame> scheduleGames = schedulesByDate.computeIfAbsent(
					eventMatch.get().eventDate(),
					this.mlbScheduleClient::fetchGames
			);
			Optional<MlbScheduleClient.MlbScheduledGame> scheduledGame = findScheduledGame(
					scheduleGames,
					oddsEvent.homeTeam(),
					oddsEvent.awayTeam()
			);

			if (scheduledGame.isEmpty()) {
				unmatchedEvents++;
				continue;
			}

			OddsRefreshRepository.EventMatch matchedEvent = eventMatch.get();
			matchedEvents++;
			if (marketId < 0) {
				marketId = this.repository.upsertPitcherStrikeoutsMarket(matchedEvent.sportId());
			}

			Map<Long, Set<String>> retainedOffersBySportsbook = new HashMap<>();
			for (OddsApiClient.PitcherStrikeoutOffer offer : oddsEvent.offers()) {
				if (offer.line() == null
						|| offer.americanPrice() == null
						|| !hasText(offer.pitcherName())
						|| !hasText(offer.side())
						|| !PitcherStrikeoutOfferSupport.isSupportedSide(offer.side())) {
					continue;
				}

				Optional<ResolvedPitcher> resolvedPitcher = resolvePitcher(scheduledGame.get(), offer.pitcherName());
				if (resolvedPitcher.isEmpty()) {
					unmatchedPitchers++;
					continue;
				}

				ResolvedPitcher pitcher = resolvedPitcher.get();
				long sportsbookId = this.repository.upsertSportsbook(new OddsRefreshRepository.SportsbookUpsert(
						offer.sportsbookKey().toUpperCase(Locale.US),
						offer.sportsbookKey(),
						offer.sportsbookDisplayName(),
						"US"
				));
				sportsbookIdsTouched.add(sportsbookId);

				String externalRef = "mlbam_player:" + pitcher.mlbamPlayerId();
				long playerId = this.repository.upsertPlayer(new OddsRefreshRepository.PlayerUpsert(
						matchedEvent.sportId(),
						externalRef,
						pitcher.fullName()
				));
				playerRefsTouched.add(externalRef);

				long eventParticipantId = this.repository.ensurePlayerEventParticipant(
						new OddsRefreshRepository.PlayerEventParticipantUpsert(
								matchedEvent.eventId(),
								playerId,
								pitcher.home() ? "STARTING_PITCHER_HOME" : "STARTING_PITCHER_AWAY",
								pitcher.home(),
								!pitcher.home(),
								pitcher.home() ? 11 : 10
						)
				);
				participantKeysTouched.add(matchedEvent.eventId() + "|" + playerId);

				String normalizedSide = PitcherStrikeoutOfferSupport.normalizeSide(offer.side());
				String sourceOfferId = buildSourceOfferId(
						oddsEvent.sourceEventId(),
						externalRef,
						normalizedSide,
						offer.line(),
						offer.sportsbookKey()
				);

				this.repository.upsertOffer(new OddsRefreshRepository.OfferUpsert(
						sportsbookId,
						matchedEvent.eventId(),
						marketId,
						eventParticipantId,
						offer.line(),
						offer.americanPrice(),
						PitcherStrikeoutOfferSupport.toDecimalPrice(offer.americanPrice()),
						PitcherStrikeoutOfferSupport.buildSelectionLabel(pitcher.fullName(), normalizedSide, offer.line()),
						normalizedSide.toUpperCase(Locale.US),
						"PROP",
						offer.availableAt() == null ? OffsetDateTime.now() : offer.availableAt(),
						false,
						sourceOfferId
				));
				offersUpserted++;
				retainedOffersBySportsbook.computeIfAbsent(sportsbookId, ignored -> new HashSet<>())
						.add(sourceOfferId);
			}

			for (Map.Entry<Long, Set<String>> entry : retainedOffersBySportsbook.entrySet()) {
				offersRemoved += this.repository.deleteMissingOffers(
						entry.getKey(),
						matchedEvent.eventId(),
						marketId,
						entry.getValue()
				);
			}
		}

		Map<String, Object> details = new java.util.LinkedHashMap<>();
		details.put("marketKey", MARKET_KEY);
		details.put("eventsExamined", oddsEvents.size());
		details.put("matchedEvents", matchedEvents);
		details.put("unmatchedEvents", unmatchedEvents);
		details.put("unmatchedPitchers", unmatchedPitchers);
		details.put("sportsbooksTouched", sportsbookIdsTouched.size());
		details.put("playersUpserted", playerRefsTouched.size());
		details.put("participantsEnsured", participantKeysTouched.size());
		details.put("offersUpserted", offersUpserted);
		details.put("offersRemoved", offersRemoved);
		details.put("bootstrapSportsData", baselineSportsData.details());
		details.put("oddsEventSchedule", Map.of(
				"datesRequested", oddsEventScheduleRefresh.datesRequested(),
				"datesFetched", oddsEventScheduleRefresh.datesFetched(),
				"gamesProcessed", oddsEventScheduleRefresh.gamesProcessed(),
				"teamsTouched", oddsEventScheduleRefresh.teamsTouched(),
				"probablePitchersEnsured", oddsEventScheduleRefresh.probablePitchersEnsured()
		));

		return new BackgroundJobResult(
				"MLB pitcher strikeout odds refresh completed",
				Map.copyOf(details)
		);
	}

	private static Optional<MlbScheduleClient.MlbScheduledGame> findScheduledGame(
			List<MlbScheduleClient.MlbScheduledGame> scheduleGames,
			String homeTeam,
			String awayTeam
	) {
		String normalizedHome = normalizeName(homeTeam);
		String normalizedAway = normalizeName(awayTeam);
		return scheduleGames.stream()
				.filter(game -> game.homeTeam() != null && normalizeName(game.homeTeam().fullName()).equals(normalizedHome))
				.filter(game -> game.awayTeam() != null && normalizeName(game.awayTeam().fullName()).equals(normalizedAway))
				.findFirst();
	}

	private static Optional<ResolvedPitcher> resolvePitcher(
			MlbScheduleClient.MlbScheduledGame game,
			String pitcherName
	) {
		String normalizedPitcherName = normalizeName(pitcherName);
		if (game.homeProbablePitcher() != null && normalizeName(game.homeProbablePitcher().fullName()).equals(normalizedPitcherName)) {
			return Optional.of(new ResolvedPitcher(
					game.homeProbablePitcher().mlbamPlayerId(),
					game.homeProbablePitcher().fullName(),
					true
			));
		}
		if (game.awayProbablePitcher() != null && normalizeName(game.awayProbablePitcher().fullName()).equals(normalizedPitcherName)) {
			return Optional.of(new ResolvedPitcher(
					game.awayProbablePitcher().mlbamPlayerId(),
					game.awayProbablePitcher().fullName(),
					false
			));
		}
		return Optional.empty();
	}

	private static String buildSourceOfferId(
			String eventSourceId,
			String externalRef,
			String side,
			BigDecimal line,
			String sportsbookKey
	) {
		return String.join(
				"|",
				"MLB",
				MARKET_KEY,
				eventSourceId,
				externalRef,
				side,
				line.stripTrailingZeros().toPlainString(),
				sportsbookKey
		);
	}

	private static String normalizeName(String value) {
		return value == null ? "" : Normalizer.normalize(value.trim(), Normalizer.Form.NFD)
				.replaceAll("\\p{M}", "")
				.toLowerCase(Locale.US)
				.replace(".", "")
				.replace("'", "")
				.replace("-", " ")
				.replaceAll("\\s+", " ");
	}

	private static boolean hasText(String value) {
		return value != null && !value.isBlank();
	}

	private record ResolvedPitcher(
			long mlbamPlayerId,
			String fullName,
			boolean home
	) {
	}
}
