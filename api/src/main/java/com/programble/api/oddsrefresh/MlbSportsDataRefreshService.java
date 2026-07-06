package com.programble.api.oddsrefresh;

import java.time.LocalDate;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.programble.api.jobs.BackgroundJobResult;
import org.springframework.stereotype.Service;

@Service
public class MlbSportsDataRefreshService {

	private final MlbScheduleClient mlbScheduleClient;
	private final OddsRefreshRepository repository;
	private final MlbTrackedOfferBackfillService trackedOfferBackfillService;

	public MlbSportsDataRefreshService(
			MlbScheduleClient mlbScheduleClient,
			OddsRefreshRepository repository,
			MlbTrackedOfferBackfillService trackedOfferBackfillService
	) {
		this.mlbScheduleClient = mlbScheduleClient;
		this.repository = repository;
		this.trackedOfferBackfillService = trackedOfferBackfillService;
	}

	public BackgroundJobResult refresh() {
		MlbTrackedOfferBackfillService.TrackedOfferDataset trackedDataset = this.trackedOfferBackfillService.loadTrackedOffers();
		Set<LocalDate> dates = new LinkedHashSet<>(trackedDataset.dates());
		LocalDate today = LocalDate.now(MlbTime.EASTERN_ZONE);
		dates.add(today);
		dates.add(today.plusDays(1));

		ScheduleRefreshSummary scheduleSummary = ensureScheduleForDates(dates);
		MlbTrackedOfferBackfillService.TrackedOfferBackfillSummary backfillSummary = this.trackedOfferBackfillService.backfill(trackedDataset);

		return new BackgroundJobResult(
				"MLB sports data refresh completed",
				Map.of(
						"datesRequested", scheduleSummary.datesRequested(),
						"datesFetched", scheduleSummary.datesFetched(),
						"gamesProcessed", scheduleSummary.gamesProcessed(),
						"teamsTouched", scheduleSummary.teamsTouched(),
						"probablePitchersEnsured", scheduleSummary.probablePitchersEnsured(),
						"trackedOffersImported", backfillSummary.offersImported(),
						"trackedOffersSkipped", backfillSummary.offersSkipped(),
						"trackedOfferRowsLoaded", trackedDataset.totalOffers()
				)
		);
	}

	public ScheduleRefreshSummary ensureScheduleForOddsEvents(List<OddsApiClient.MlbPitcherStrikeoutEvent> oddsEvents) {
		Set<LocalDate> dates = new LinkedHashSet<>();
		for (OddsApiClient.MlbPitcherStrikeoutEvent event : oddsEvents) {
			if (event.commenceTime() == null) {
				continue;
			}
			dates.add(event.commenceTime().toLocalDate());
			dates.add(MlbTime.toEasternDate(event.commenceTime()));
		}
		return ensureScheduleForDates(dates);
	}

	public ScheduleRefreshSummary ensureScheduleForDates(Set<LocalDate> dates) {
		if (dates.isEmpty()) {
			return new ScheduleRefreshSummary(0, 0, 0, 0, 0);
		}

		long sportId = this.repository.ensureMlbSport();
		long competitionId = this.repository.ensureMlbCompetition(sportId);

		int datesFetched = 0;
		int gamesProcessed = 0;
		int probablePitchersEnsured = 0;
		Set<String> teamRefsTouched = new LinkedHashSet<>();

		for (LocalDate date : dates.stream().sorted().toList()) {
			List<MlbScheduleClient.MlbScheduledGame> games = this.mlbScheduleClient.fetchGames(date);
			datesFetched++;

			for (MlbScheduleClient.MlbScheduledGame game : games) {
				if (game.homeTeam() == null || game.awayTeam() == null || game.scheduledStart() == null) {
					continue;
				}

				long homeTeamId = this.repository.upsertTeam(buildTeamUpsert(sportId, game.homeTeam()));
				long awayTeamId = this.repository.upsertTeam(buildTeamUpsert(sportId, game.awayTeam()));
				teamRefsTouched.add("mlb_stats_api_team:" + game.homeTeam().mlbamTeamId());
				teamRefsTouched.add("mlb_stats_api_team:" + game.awayTeam().mlbamTeamId());

				this.repository.ensureCompetitionTeam(new OddsRefreshRepository.CompetitionTeamUpsert(
						competitionId,
						homeTeamId,
						"mlb_stats_api_team:" + game.homeTeam().mlbamTeamId()
				));
				this.repository.ensureCompetitionTeam(new OddsRefreshRepository.CompetitionTeamUpsert(
						competitionId,
						awayTeamId,
						"mlb_stats_api_team:" + game.awayTeam().mlbamTeamId()
				));

				LocalDate easternGameDate = MlbTime.toEasternDate(game.scheduledStart());
				OddsRefreshRepository.EventMatch eventMatch = this.repository.upsertMlbEvent(
						new OddsRefreshRepository.EventUpsert(
								sportId,
								competitionId,
								"mlb_stats_api_game:" + game.gamePk(),
								buildEventSlug(game, easternGameDate),
								"GAME",
								game.status(),
								String.valueOf(easternGameDate.getYear()),
								roundLabel(game.gameType()),
								game.scheduledStart(),
								true,
								game.venue() == null ? null : game.venue().name(),
								game.venue() == null ? null : game.venue().city(),
								game.venue() == null ? "US" : game.venue().countryCode(),
								homeTeamId,
								awayTeamId,
								game.homeTeam().abbreviation(),
								game.awayTeam().abbreviation()
						)
				);

				this.repository.ensureTeamEventParticipant(new OddsRefreshRepository.TeamEventParticipantUpsert(
						eventMatch.eventId(),
						homeTeamId,
						"HOME",
						true,
						false,
						1
				));
				this.repository.ensureTeamEventParticipant(new OddsRefreshRepository.TeamEventParticipantUpsert(
						eventMatch.eventId(),
						awayTeamId,
						"AWAY",
						false,
						true,
						2
				));

				probablePitchersEnsured += ensureProbablePitcher(sportId, eventMatch, game.homeProbablePitcher(), true);
				probablePitchersEnsured += ensureProbablePitcher(sportId, eventMatch, game.awayProbablePitcher(), false);
				gamesProcessed++;
			}
		}

		return new ScheduleRefreshSummary(
				dates.size(),
				datesFetched,
				gamesProcessed,
				teamRefsTouched.size(),
				probablePitchersEnsured
		);
	}

	private int ensureProbablePitcher(
			long sportId,
			OddsRefreshRepository.EventMatch eventMatch,
			MlbScheduleClient.ScheduledPitcher pitcher,
			boolean home
	) {
		if (pitcher == null || pitcher.fullName() == null || pitcher.fullName().isBlank()) {
			return 0;
		}

		long playerId = this.repository.upsertPlayer(new OddsRefreshRepository.PlayerUpsert(
				sportId,
				"mlbam_player:" + pitcher.mlbamPlayerId(),
				pitcher.fullName()
		));
		this.repository.ensurePlayerEventParticipant(
				new OddsRefreshRepository.PlayerEventParticipantUpsert(
						eventMatch.eventId(),
						playerId,
						home ? "STARTING_PITCHER_HOME" : "STARTING_PITCHER_AWAY",
						home,
						!home,
						home ? 11 : 10
				)
		);
		return 1;
	}

	private static OddsRefreshRepository.TeamUpsert buildTeamUpsert(long sportId, MlbScheduleClient.ScheduledTeam team) {
		String code = MlbTeamMappings.canonicalCode(team.abbreviation());
		return new OddsRefreshRepository.TeamUpsert(
				sportId,
				"mlb_stats_api_team:" + team.mlbamTeamId(),
				code,
				code,
				team.fullName(),
				team.city(),
				"US"
		);
	}

	private static String buildEventSlug(MlbScheduleClient.MlbScheduledGame game, LocalDate easternGameDate) {
		return MlbTeamMappings.slugify(game.awayTeam().fullName())
				+ "-"
				+ MlbTeamMappings.slugify(game.homeTeam().fullName())
				+ "-"
				+ easternGameDate
				+ "-"
				+ game.gamePk();
	}

	private static String roundLabel(String gameType) {
		if (gameType == null || gameType.isBlank()) {
			return "Regular Season";
		}
		return switch (gameType.trim().toUpperCase()) {
			case "R" -> "Regular Season";
			case "F", "D", "L", "W", "C" -> "Postseason";
			case "S" -> "Spring Training";
			default -> "Regular Season";
		};
	}

	public record ScheduleRefreshSummary(
			int datesRequested,
			int datesFetched,
			int gamesProcessed,
			int teamsTouched,
			int probablePitchersEnsured
	) {
	}
}
