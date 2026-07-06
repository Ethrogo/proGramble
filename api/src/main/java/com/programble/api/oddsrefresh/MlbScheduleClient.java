package com.programble.api.oddsrefresh;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;

public interface MlbScheduleClient {

	List<MlbScheduledGame> fetchGames(LocalDate date);

	record MlbScheduledGame(
			String gamePk,
			OffsetDateTime scheduledStart,
			String gameType,
			String status,
			ScheduledVenue venue,
			ScheduledTeam homeTeam,
			ScheduledTeam awayTeam,
			ScheduledPitcher homeProbablePitcher,
			ScheduledPitcher awayProbablePitcher
	) {
	}

	record ScheduledTeam(
			long mlbamTeamId,
			String fullName,
			String abbreviation,
			String city
	) {
	}

	record ScheduledVenue(
			String name,
			String city,
			String countryCode
	) {
	}

	record ScheduledPitcher(
			long mlbamPlayerId,
			String fullName
	) {
	}
}
