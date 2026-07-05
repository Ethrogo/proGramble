package com.programble.api.oddsrefresh;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.BDDMockito.given;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;

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

	@BeforeEach
	void seedCatalog() {
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

		this.jdbcTemplate.update(
				"insert into sports (id, code, slug, name, is_active) values (?, ?, ?, ?, ?)",
				1L, "MLB", "mlb", "Major League Baseball", true
		);
		this.jdbcTemplate.update(
				"insert into competitions (id, sport_id, code, slug, name, competition_type, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				11L, 1L, "MLB", "mlb", "MLB Regular Season", "TEAM", true
		);
		this.jdbcTemplate.update(
				"insert into teams (id, sport_id, slug, code, short_name, full_name, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				101L, 1L, "boston-red-sox", "BOS", "BOS", "Boston Red Sox", true
		);
		this.jdbcTemplate.update(
				"insert into teams (id, sport_id, slug, code, short_name, full_name, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				102L, 1L, "new-york-yankees", "NYY", "NYY", "New York Yankees", true
		);
		this.jdbcTemplate.update(
				"""
				insert into events (
				    id, sport_id, competition_id, slug, external_ref, event_type, status, season_label, round_label,
				    scheduled_start, start_time_confirmed, venue_name, venue_city, venue_country_code
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				1001L, 1L, 11L, "red-sox-yankees-2026-06-13", "mlb-20260613-bos-nyy", "GAME", "SCHEDULED", "2026", "Regular Season",
				"2026-06-13T23:10:00Z", true, "Fenway Park", "Boston", "US"
		);
		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, team_id, role_code, sort_order, is_home, is_away) values (?, ?, ?, ?, ?, ?, ?)",
				9001L, 1001L, 101L, "HOME", 1, true, false
		);
		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, team_id, role_code, sort_order, is_home, is_away) values (?, ?, ?, ?, ?, ?, ?)",
				9002L, 1001L, 102L, "AWAY", 2, false, true
		);
	}

	@Test
	void refreshesAndUpsertsPitcherStrikeoutOffers() {
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
						"Boston Red Sox",
						"New York Yankees",
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

		assertThat(count("select count(*) from sportsbooks")).isEqualTo(2);
		assertThat(count("select count(*) from markets where code = 'PITCHER_STRIKEOUTS'")).isEqualTo(1);
		assertThat(count("select count(*) from players where external_ref = 'mlbam_player:555'")).isEqualTo(1);
		assertThat(count("select count(*) from event_participants where event_id = 1001 and player_id is not null")).isEqualTo(1);
		assertThat(count("select count(*) from offers where event_id = 1001")).isEqualTo(3);
		assertThat(count("select count(*) from offers where side_code = 'OVER'")).isEqualTo(2);
		assertThat(count("select count(*) from offers where side_code = 'UNDER'")).isEqualTo(1);
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
						"Boston Red Sox",
						"New York Yankees",
						new MlbScheduleClient.ScheduledPitcher(555L, "Garrett Crochet"),
						new MlbScheduleClient.ScheduledPitcher(777L, "Max Fried")
				)
		));

		this.refreshService.refresh();
		BackgroundJobResult result = this.refreshService.refresh();

		assertThat(result.details()).containsEntry("offersRemoved", 1);
		assertThat(count("select count(*) from offers where event_id = 1001")).isEqualTo(1);
		assertThat(count("select count(*) from offers where side_code = 'UNDER'")).isZero();
		assertThat(singleInteger(
				"select price_american from offers where event_id = 1001 and side_code = 'OVER'"
		)).isEqualTo(-125);
	}

	private long count(String sql) {
		return this.jdbcTemplate.queryForObject(sql, Long.class);
	}

	private Integer singleInteger(String sql) {
		return this.jdbcTemplate.queryForObject(sql, Integer.class);
	}
}
