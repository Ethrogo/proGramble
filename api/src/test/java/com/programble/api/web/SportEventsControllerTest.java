package com.programble.api.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest(properties = "programble.environment=test")
@AutoConfigureMockMvc
class SportEventsControllerTest {

	@Autowired
	private MockMvc mockMvc;

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
