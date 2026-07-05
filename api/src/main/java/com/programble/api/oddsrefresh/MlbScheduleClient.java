package com.programble.api.oddsrefresh;

import java.time.LocalDate;
import java.util.List;

public interface MlbScheduleClient {

	List<MlbScheduledGame> fetchGames(LocalDate date);

	record MlbScheduledGame(
			String gamePk,
			String homeTeam,
			String awayTeam,
			ScheduledPitcher homeProbablePitcher,
			ScheduledPitcher awayProbablePitcher
	) {
	}

	record ScheduledPitcher(
			long mlbamPlayerId,
			String fullName
	) {
	}
}
