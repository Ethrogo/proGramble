package com.programble.api.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class MarketOffersControllerTest {

	@Autowired
	private MockMvc mockMvc;

	@Autowired
	private JdbcTemplate jdbcTemplate;

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
				101L, 1L, "new-york-yankees", "NYY", "NYY", "New York Yankees", true
		);
		this.jdbcTemplate.update(
				"insert into teams (id, sport_id, slug, code, short_name, full_name, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				102L, 1L, "boston-red-sox", "BOS", "BOS", "Boston Red Sox", true
		);
		this.jdbcTemplate.update(
				"insert into players (id, sport_id, slug, first_name, last_name, display_name, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				301L, 1L, "aaron-judge", "Aaron", "Judge", "Aaron Judge", true
		);
		this.jdbcTemplate.update(
				"insert into players (id, sport_id, slug, first_name, last_name, display_name, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				302L, 1L, "rafael-devers", "Rafael", "Devers", "Rafael Devers", true
		);

		this.jdbcTemplate.update(
				"""
				insert into events (
				    id, sport_id, competition_id, slug, external_ref, event_type, status, season_label, round_label,
				    scheduled_start, start_time_confirmed, venue_name, venue_city, venue_country_code
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				1001L, 1L, 11L, "yankees-red-sox-2026-06-13", "mlb-20260613-nyy-bos", "GAME", "SCHEDULED", "2026", "Regular Season",
				"2026-06-13T23:10:00Z", true, "Fenway Park", "Boston", "US"
		);

		this.jdbcTemplate.update(
				"insert into sportsbooks (id, code, slug, display_name, region_code, is_active) values (?, ?, ?, ?, ?, ?)",
				601L, "DK", "draftkings", "DraftKings", "US", true
		);
		this.jdbcTemplate.update(
				"insert into sportsbooks (id, code, slug, display_name, region_code, is_active) values (?, ?, ?, ?, ?, ?)",
				602L, "FD", "fanduel", "FanDuel", "US", true
		);

		this.jdbcTemplate.update(
				"""
				insert into markets (
				    id, sport_id, competition_id, code, slug, display_name, market_scope, stat_type, period_type,
				    allows_over_under, allows_binary_outcome
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				501L, 1L, null, "MONEYLINE", "moneyline", "Moneyline", "EVENT", null, "FULL_EVENT", false, true
		);
		this.jdbcTemplate.update(
				"""
				insert into markets (
				    id, sport_id, competition_id, code, slug, display_name, market_scope, stat_type, period_type,
				    allows_over_under, allows_binary_outcome
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				502L, 1L, null, "PLAYER_HITS", "player-hits", "Player Hits", "PLAYER", "HITS", "FULL_EVENT", true, false
		);
		this.jdbcTemplate.update(
				"""
				insert into markets (
				    id, sport_id, competition_id, code, slug, display_name, market_scope, stat_type, period_type,
				    allows_over_under, allows_binary_outcome
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				503L, 1L, null, "PLAYER_HOME_RUNS", "player-home-runs", "Player Home Runs", "PLAYER", "HOME_RUNS", "FULL_EVENT", true, false
		);

		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, team_id, role_code, sort_order, is_home, is_away) values (?, ?, ?, ?, ?, ?, ?)",
				9001L, 1001L, 101L, "AWAY", 1, false, true
		);
		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, team_id, role_code, sort_order, is_home, is_away) values (?, ?, ?, ?, ?, ?, ?)",
				9002L, 1001L, 102L, "HOME", 2, true, false
		);
		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, player_id, role_code, sort_order) values (?, ?, ?, ?, ?)",
				9101L, 1001L, 301L, "BATTER_ONE", 3
		);
		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, player_id, role_code, sort_order) values (?, ?, ?, ?, ?)",
				9102L, 1001L, 302L, "BATTER_TWO", 4
		);

		this.jdbcTemplate.update(
				"""
				insert into offers (
				    id, sportsbook_id, event_id, market_id, event_participant_id, line_value, price_american, price_decimal,
				    selection_label, side_code, outcome_type, available_at, is_live, source_offer_id
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				7001L, 601L, 1001L, 501L, null, null, -125, 1.8000, "Boston Red Sox", "HOME", "MONEYLINE", "2026-06-13T12:00:00Z", false, "dk-ml-home"
		);
		this.jdbcTemplate.update(
				"""
				insert into offers (
				    id, sportsbook_id, event_id, market_id, event_participant_id, line_value, price_american, price_decimal,
				    selection_label, side_code, outcome_type, available_at, is_live, source_offer_id
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				7002L, 602L, 1001L, 501L, null, null, 110, 2.1000, "New York Yankees", "AWAY", "MONEYLINE", "2026-06-13T12:01:00Z", false, "fd-ml-away"
		);
		this.jdbcTemplate.update(
				"""
				insert into offers (
				    id, sportsbook_id, event_id, market_id, event_participant_id, line_value, price_american, price_decimal,
				    selection_label, side_code, outcome_type, available_at, is_live, source_offer_id
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				7003L, 601L, 1001L, 502L, 9101L, 1.5, -105, 1.9524, "Aaron Judge Over 1.5 Hits", "OVER", "PROP", "2026-06-13T12:02:00Z", false, "dk-judge-hits-over"
		);
		this.jdbcTemplate.update(
				"""
				insert into offers (
				    id, sportsbook_id, event_id, market_id, event_participant_id, line_value, price_american, price_decimal,
				    selection_label, side_code, outcome_type, available_at, is_live, source_offer_id
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				7004L, 602L, 1001L, 502L, 9101L, 1.5, -115, 1.8696, "Aaron Judge Under 1.5 Hits", "UNDER", "PROP", "2026-06-13T12:03:00Z", true, "fd-judge-hits-under"
		);
		this.jdbcTemplate.update(
				"""
				insert into offers (
				    id, sportsbook_id, event_id, market_id, event_participant_id, line_value, price_american, price_decimal,
				    selection_label, side_code, outcome_type, available_at, is_live, source_offer_id
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				7005L, 601L, 1001L, 503L, 9101L, 0.5, 240, 3.4000, "Aaron Judge Over 0.5 Home Runs", "OVER", "PROP", "2026-06-13T12:04:00Z", false, "dk-judge-hr-over"
		);
	}

	@Test
	void listsOffersForAnEvent() throws Exception {
		mockMvc.perform(get("/api/v1/events/1001/offers"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.event.id").value(1001))
				.andExpect(jsonPath("$.event.sport.code").value("MLB"))
				.andExpect(jsonPath("$.count").value(5))
				.andExpect(jsonPath("$.offers[0].market.code").value("MONEYLINE"))
				.andExpect(jsonPath("$.offers[0].participant").doesNotExist())
				.andExpect(jsonPath("$.offers[2].participant.playerId").value(301))
				.andExpect(jsonPath("$.offers[2].sportsbook.code").value("DK"));
	}

	@Test
	void filtersEventOffersBySportsbook() throws Exception {
		mockMvc.perform(get("/api/v1/events/1001/offers").param("sportsbook", "draftkings"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.filters.sportsbook").value("draftkings"))
				.andExpect(jsonPath("$.count").value(3))
				.andExpect(jsonPath("$.offers[0].sportsbook.slug").value("draftkings"))
				.andExpect(jsonPath("$.offers[1].sportsbook.slug").value("draftkings"))
				.andExpect(jsonPath("$.offers[2].sportsbook.slug").value("draftkings"));
	}

	@Test
	void filtersEventOffersByPlayerAndMarketType() throws Exception {
		mockMvc.perform(get("/api/v1/events/1001/offers")
						.param("playerId", "301")
						.param("marketType", "hits"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.filters.playerId").value(301))
				.andExpect(jsonPath("$.filters.marketType").value("hits"))
				.andExpect(jsonPath("$.count").value(2))
				.andExpect(jsonPath("$.offers[0].participant.playerId").value(301))
				.andExpect(jsonPath("$.offers[0].market.statType").value("HITS"))
				.andExpect(jsonPath("$.offers[1].participant.playerId").value(301))
				.andExpect(jsonPath("$.offers[1].market.statType").value("HITS"));
	}

	@Test
	void listsOffersForAPlayer() throws Exception {
		mockMvc.perform(get("/api/v1/players/301/offers"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.player.id").value(301))
				.andExpect(jsonPath("$.player.displayName").value("Aaron Judge"))
				.andExpect(jsonPath("$.count").value(3))
				.andExpect(jsonPath("$.offers[0].event.id").value(1001))
				.andExpect(jsonPath("$.offers[0].participant.playerId").value(301));
	}

	@Test
	void filtersPlayerOffersBySportsbookAndMarketType() throws Exception {
		mockMvc.perform(get("/api/v1/players/301/offers")
						.param("sportsbook", "DK")
						.param("marketType", "player-home-runs"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.count").value(1))
				.andExpect(jsonPath("$.offers[0].sportsbook.code").value("DK"))
				.andExpect(jsonPath("$.offers[0].market.slug").value("player-home-runs"));
	}

	@Test
	void returnsNotFoundForUnknownEventOrPlayer() throws Exception {
		mockMvc.perform(get("/api/v1/events/999999/offers"))
				.andExpect(status().isNotFound());

		mockMvc.perform(get("/api/v1/players/999999/offers"))
				.andExpect(status().isNotFound());
	}
}
