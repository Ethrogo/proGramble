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
class SportEventsControllerTest {

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
				"insert into sports (id, code, slug, name, is_active) values (?, ?, ?, ?, ?)",
				3L, "TENNIS", "tennis", "Tennis", true
		);
		this.jdbcTemplate.update(
				"insert into sports (id, code, slug, name, is_active) values (?, ?, ?, ?, ?)",
				4L, "GOLF", "golf", "Golf", true
		);

		this.jdbcTemplate.update(
				"insert into competitions (id, sport_id, code, slug, name, competition_type, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				11L, 1L, "MLB", "mlb", "MLB Regular Season", "TEAM", true
		);
		this.jdbcTemplate.update(
				"insert into competitions (id, sport_id, code, slug, name, competition_type, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				31L, 3L, "ATP-500", "atp-500", "ATP 500", "INDIVIDUAL", true
		);
		this.jdbcTemplate.update(
				"insert into competitions (id, sport_id, code, slug, name, competition_type, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				41L, 4L, "PGA", "pga", "PGA Tour", "INDIVIDUAL", true
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
				201L, 3L, "carlos-alcaraz", "Carlos", "Alcaraz", "Carlos Alcaraz", true
		);
		this.jdbcTemplate.update(
				"insert into players (id, sport_id, slug, first_name, last_name, display_name, is_active) values (?, ?, ?, ?, ?, ?, ?)",
				202L, 3L, "jannik-sinner", "Jannik", "Sinner", "Jannik Sinner", true
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
				"""
				insert into events (
				    id, sport_id, competition_id, slug, external_ref, event_type, status, season_label, round_label,
				    scheduled_start, start_time_confirmed, venue_name, venue_city, venue_country_code
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				1002L, 1L, 11L, "dodgers-giants-2026-06-14", "mlb-20260614-lad-sf", "GAME", "SCHEDULED", "2026", "Regular Season",
				"2026-06-14T20:10:00Z", true, "Oracle Park", "San Francisco", "US"
		);
		this.jdbcTemplate.update(
				"""
				insert into events (
				    id, sport_id, competition_id, slug, external_ref, event_type, status, season_label, round_label,
				    scheduled_start, start_time_confirmed, venue_name, venue_city, venue_country_code
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				3001L, 3L, 31L, "alcaraz-sinner-2026-06-13", "atp-20260613-car-sin", "MATCH", "SCHEDULED", "2026", "Quarterfinal",
				"2026-06-13T13:00:00Z", true, "The Queen's Club", "London", "GB"
		);
		this.jdbcTemplate.update(
				"""
				insert into events (
				    id, sport_id, competition_id, slug, external_ref, event_type, status, season_label, round_label,
				    scheduled_start, start_time_confirmed, venue_name, venue_city, venue_country_code
				) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				4001L, 4L, 41L, "us-open-2026", "pga-2026-us-open", "TOURNAMENT", "SCHEDULED", "2026", "Week 24",
				"2026-06-13T12:00:00Z", false, "Oakmont Country Club", "Oakmont", "US"
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
				"insert into event_participants (id, event_id, player_id, role_code, seed_value, sort_order) values (?, ?, ?, ?, ?, ?)",
				9003L, 3001L, 201L, "PLAYER_ONE", 1, 1
		);
		this.jdbcTemplate.update(
				"insert into event_participants (id, event_id, player_id, role_code, seed_value, sort_order) values (?, ?, ?, ?, ?, ?)",
				9004L, 3001L, 202L, "PLAYER_TWO", 2, 2
		);
	}

	@Test
	void listsUpcomingEventsBySportAndDate() throws Exception {
		mockMvc.perform(get("/api/v1/sports/MLB/events").param("date", "2026-06-13"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.sport.code").value("MLB"))
				.andExpect(jsonPath("$.sport.slug").value("mlb"))
				.andExpect(jsonPath("$.date").value("2026-06-13"))
				.andExpect(jsonPath("$.count").value(1))
				.andExpect(jsonPath("$.events[0].id").value(1001))
				.andExpect(jsonPath("$.events[0].eventType").value("GAME"))
				.andExpect(jsonPath("$.events[0].competition.slug").value("mlb"))
				.andExpect(jsonPath("$.events[0].venue.name").value("Fenway Park"))
				.andExpect(jsonPath("$.events[0].participants[0].type").value("TEAM"))
				.andExpect(jsonPath("$.events[0].participants[0].shortName").value("NYY"));
	}

	@Test
	void returnsEmptySlateWhenSportExistsButNoEventsMatchTheDate() throws Exception {
		mockMvc.perform(get("/api/v1/sports/golf/events").param("date", "2026-06-14"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.sport.code").value("GOLF"))
				.andExpect(jsonPath("$.date").value("2026-06-14"))
				.andExpect(jsonPath("$.count").value(0))
				.andExpect(jsonPath("$.events").isArray())
				.andExpect(jsonPath("$.events").isEmpty());
	}

	@Test
	void exposesEventDetailAcrossDifferentEventTypes() throws Exception {
		mockMvc.perform(get("/api/v1/events/3001"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.id").value(3001))
				.andExpect(jsonPath("$.sport.code").value("TENNIS"))
				.andExpect(jsonPath("$.competition.competitionType").value("INDIVIDUAL"))
				.andExpect(jsonPath("$.eventType").value("MATCH"))
				.andExpect(jsonPath("$.roundLabel").value("Quarterfinal"))
				.andExpect(jsonPath("$.participants[0].displayName").value("Carlos Alcaraz"))
				.andExpect(jsonPath("$.participants[0].seedValue").value(1));
	}

	@Test
	void returnsNotFoundForUnknownSport() throws Exception {
		mockMvc.perform(get("/api/v1/sports/soccer/events").param("date", "2026-06-13"))
				.andExpect(status().isNotFound());
	}

	@Test
	void returnsNotFoundForUnknownEvent() throws Exception {
		mockMvc.perform(get("/api/v1/events/999999"))
				.andExpect(status().isNotFound());
	}
}
