package com.programble.api.oddsrefresh;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.BDDMockito.given;
import static org.mockito.ArgumentMatchers.any;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.programble.api.jobs.BackgroundJobResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@ActiveProfiles("test")
class MlbPitcherStrikeoutsRefreshServiceTest {

	@Autowired
	private MlbPitcherStrikeoutsRefreshService refreshService;

	@Autowired
	private JdbcTemplate jdbcTemplate;

	@MockBean
	private OddsApiClient oddsApiClient;

	@MockBean
	private MlbScheduleClient mlbScheduleClient;

	@MockBean
	private MlbTrackedOfferBackfillService trackedOfferBackfillService;

	@BeforeEach
	void clearCatalog() {
		this.jdbcTemplate.execute("delete from offers");
		this.jdbcTemplate.execute("delete from event_participants");
		this.jdbcTemplate.execute("delete from markets");
		this.jdbcTemplate.execute("delete from sportsbooks");
		this.jdbcTemplate.execute("delete from events");
		this.jdbcTemplate.execute("delete from team_players");
		this.jdbcTemplate.execute("delete from competition_teams");
		this.jdbcTemplate.execute("delete from players");
		this.jdbcTemplate.execute("delete from teams");
		this.jdbcTemplate.execute("delete from competitions");
		this.jdbcTemplate.execute("delete from sports");

		given(this.trackedOfferBackfillService.loadTrackedOffers()).willReturn(
				new MlbTrackedOfferBackfillService.TrackedOfferDataset(List.of(), Set.of(), 0, 0, 0)
		);
		given(this.trackedOfferBackfillService.backfill(any())).willReturn(
				new MlbTrackedOfferBackfillService.TrackedOfferBackfillSummary(0, 0, 0, 0)
		);
	}

	@Test
	void ingestsScheduleDataThenUpsertsPitcherStrikeoutOffers() {
		given(this.oddsApiClient.fetchMlbPitcherStrikeoutEvents()).willReturn(List.of(
				new OddsApiClient.MlbPitcherStrikeoutEvent(
						"odds-event-1",
						OffsetDateTime.parse("2026-06-13T23:10:00Z"),
						"Boston Red Sox",
						"New York Yankees",
						List.of(
								new OddsApiClient.PitcherStrikeoutOffer(
										"draftkings",
										"DraftKings",
										OffsetDateTime.parse("2026-06-13T18:00:00Z"),
										"Garrett Crochet",
										"Over",
										new BigDecimal("6.5"),
										-120
								),
								new OddsApiClient.PitcherStrikeoutOffer(
										"draftkings",
										"DraftKings",
										OffsetDateTime.parse("2026-06-13T18:00:00Z"),
										"Garrett Crochet",
										"Under",
										new BigDecimal("6.5"),
										100
								),
								new OddsApiClient.PitcherStrikeoutOffer(
										"fanduel",
										"FanDuel",
										OffsetDateTime.parse("2026-06-13T18:01:00Z"),
										"Garrett Crochet",
										"Over",
										new BigDecimal("6.5"),
										-115
								)
						)
				),
				new OddsApiClient.MlbPitcherStrikeoutEvent(
						"odds-event-2",
						OffsetDateTime.parse("2026-06-14T00:10:00Z"),
						"Los Angeles Dodgers",
						"San Diego Padres",
						List.of()
				)
		));
		given(this.mlbScheduleClient.fetchGames(LocalDate.of(2026, 6, 13))).willReturn(List.of(
				new MlbScheduleClient.MlbScheduledGame(
						"12345",
						OffsetDateTime.parse("2026-06-13T23:10:00Z"),
						"R",
						"SCHEDULED",
						new MlbScheduleClient.ScheduledVenue("Fenway Park", "Boston", "US"),
						new MlbScheduleClient.ScheduledTeam(111L, "Boston Red Sox", "BOS", "Boston"),
						new MlbScheduleClient.ScheduledTeam(147L, "New York Yankees", "NYY", "New York"),
						new MlbScheduleClient.ScheduledPitcher(555L, "Garrett Crochet"),
						new MlbScheduleClient.ScheduledPitcher(777L, "Max Fried")
				)
		));

		BackgroundJobResult result = this.refreshService.refresh();

		assertThat(result.summary()).isEqualTo("MLB pitcher strikeout odds refresh completed");
		assertThat(result.details()).containsAllEntriesOf(Map.of(
				"marketKey", "pitcher_strikeouts",
				"eventsExamined", 2,
				"matchedEvents", 1,
				"unmatchedEvents", 1,
				"unmatchedPitchers", 0,
				"sportsbooksTouched", 2,
				"playersUpserted", 1,
				"participantsEnsured", 1,
				"offersUpserted", 3
		));

		assertThat(count("select count(*) from sports where code = 'MLB'")).isEqualTo(1);
		assertThat(count("select count(*) from competitions where code = 'MLB'")).isEqualTo(1);
		assertThat(count("select count(*) from teams where code in ('BOS', 'NYY')")).isEqualTo(2);
		assertThat(count("select count(*) from events where external_ref = 'mlb_stats_api_game:12345'")).isEqualTo(1);
		assertThat(count("select count(*) from sportsbooks")).isEqualTo(2);
		assertThat(count("select count(*) from markets where code = 'PITCHER_STRIKEOUTS'")).isEqualTo(1);
		assertThat(count("select count(*) from players where external_ref = 'mlbam_player:555'")).isEqualTo(1);
		assertThat(count("select count(*) from event_participants where team_id is not null")).isEqualTo(2);
		assertThat(count("select count(*) from event_participants where player_id is not null")).isEqualTo(2);
		assertThat(count("select count(*) from offers")).isEqualTo(3);
	}

	@Test
	void removesStaleOffersThatDisappearFromTheSameSportsbook() {
		given(this.oddsApiClient.fetchMlbPitcherStrikeoutEvents()).willReturn(
				List.of(new OddsApiClient.MlbPitcherStrikeoutEvent(
						"odds-event-1",
						OffsetDateTime.parse("2026-06-13T23:10:00Z"),
						"Boston Red Sox",
						"New York Yankees",
						List.of(
								new OddsApiClient.PitcherStrikeoutOffer(
										"draftkings",
										"DraftKings",
										OffsetDateTime.parse("2026-06-13T18:00:00Z"),
										"Garrett Crochet",
										"Over",
										new BigDecimal("6.5"),
										-120
								),
								new OddsApiClient.PitcherStrikeoutOffer(
										"draftkings",
										"DraftKings",
										OffsetDateTime.parse("2026-06-13T18:00:00Z"),
										"Garrett Crochet",
										"Under",
										new BigDecimal("6.5"),
										100
								)
						)
				)),
				List.of(new OddsApiClient.MlbPitcherStrikeoutEvent(
						"odds-event-1",
						OffsetDateTime.parse("2026-06-13T23:10:00Z"),
						"Boston Red Sox",
						"New York Yankees",
						List.of(
								new OddsApiClient.PitcherStrikeoutOffer(
										"draftkings",
										"DraftKings",
										OffsetDateTime.parse("2026-06-13T18:05:00Z"),
										"Garrett Crochet",
										"Over",
										new BigDecimal("6.5"),
										-125
								)
						)
				))
		);
		given(this.mlbScheduleClient.fetchGames(LocalDate.of(2026, 6, 13))).willReturn(List.of(
				new MlbScheduleClient.MlbScheduledGame(
						"12345",
						OffsetDateTime.parse("2026-06-13T23:10:00Z"),
						"R",
						"SCHEDULED",
						new MlbScheduleClient.ScheduledVenue("Fenway Park", "Boston", "US"),
						new MlbScheduleClient.ScheduledTeam(111L, "Boston Red Sox", "BOS", "Boston"),
						new MlbScheduleClient.ScheduledTeam(147L, "New York Yankees", "NYY", "New York"),
						new MlbScheduleClient.ScheduledPitcher(555L, "Garrett Crochet"),
						new MlbScheduleClient.ScheduledPitcher(777L, "Max Fried")
				)
		));

		this.refreshService.refresh();
		BackgroundJobResult result = this.refreshService.refresh();

		assertThat(result.details()).containsEntry("offersRemoved", 1);
		assertThat(count("select count(*) from offers")).isEqualTo(1);
		assertThat(count("select count(*) from offers where side_code = 'UNDER'")).isZero();
		assertThat(singleInteger(
				"select price_american from offers where side_code = 'OVER'"
		)).isEqualTo(-125);
	}

	private long count(String sql) {
		return this.jdbcTemplate.queryForObject(sql, Long.class);
	}

	private Integer singleInteger(String sql) {
		return this.jdbcTemplate.queryForObject(sql, Integer.class);
	}
}
