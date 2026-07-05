package com.programble.api.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import com.programble.api.jobs.BackgroundJobResult;
import com.programble.api.oddsrefresh.MlbPitcherStrikeoutsRefreshService;

import static org.mockito.BDDMockito.given;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class BackgroundJobsControllerTest {

	private static final String ADMIN_TOKEN = "test-admin-token";

	@Autowired
	private MockMvc mockMvc;

	@MockBean
	private MlbPitcherStrikeoutsRefreshService refreshService;

	@Test
	void requiresATokenForAdminRoutes() throws Exception {
		mockMvc.perform(get("/api/v1/admin/jobs"))
				.andExpect(status().isUnauthorized());
	}

	@Test
	void rejectsTheWrongTokenForAdminRoutes() throws Exception {
		mockMvc.perform(get("/api/v1/admin/jobs").header("Authorization", "Bearer wrong-token"))
				.andExpect(status().isUnauthorized());
	}

	@Test
	void listsJobsAndRunsOddsRefreshManuallyWhenAuthorized() throws Exception {
		given(this.refreshService.refresh()).willReturn(new BackgroundJobResult(
				"MLB pitcher strikeout odds refresh completed",
				Map.of(
						"marketKey", "pitcher_strikeouts",
						"offersUpserted", 2
				)
		));

		mockMvc.perform(get("/api/v1/admin/jobs").header("Authorization", bearerToken()))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.count").value(3))
				.andExpect(jsonPath("$.jobs[0].key").value("refresh-derived-site-data"))
				.andExpect(jsonPath("$.jobs[0].scheduleEnabled").value(false))
				.andExpect(jsonPath("$.jobs[1].key").value("refresh-odds"))
				.andExpect(jsonPath("$.jobs[2].key").value("refresh-sports-data"));

		mockMvc.perform(post("/api/v1/admin/jobs/refresh-odds/run").header("Authorization", bearerToken()))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.trigger").value("MANUAL"))
				.andExpect(jsonPath("$.summary").value("MLB pitcher strikeout odds refresh completed"))
				.andExpect(jsonPath("$.details.marketKey").value("pitcher_strikeouts"))
				.andExpect(jsonPath("$.details.offersUpserted").value(2))
				.andExpect(jsonPath("$.details.trigger").value("MANUAL"))
				.andExpect(jsonPath("$.job.key").value("refresh-odds"))
				.andExpect(jsonPath("$.job.totalRuns").value(1))
				.andExpect(jsonPath("$.job.successfulRuns").value(1))
				.andExpect(jsonPath("$.job.failedRuns").value(0))
				.andExpect(jsonPath("$.job.lastTrigger").value("MANUAL"));

		mockMvc.perform(get("/api/v1/admin/jobs/refresh-odds").header("Authorization", bearerToken()))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.key").value("refresh-odds"))
				.andExpect(jsonPath("$.scheduleEnabled").value(false))
				.andExpect(jsonPath("$.totalRuns").value(1))
				.andExpect(jsonPath("$.successfulRuns").value(1))
				.andExpect(jsonPath("$.lastSummary").value("MLB pitcher strikeout odds refresh completed"))
				.andExpect(jsonPath("$.lastDetails.marketKey").value("pitcher_strikeouts"));
	}

	@Test
	void returnsNotFoundForUnknownJob() throws Exception {
		mockMvc.perform(get("/api/v1/admin/jobs/missing-job").header("Authorization", bearerToken()))
				.andExpect(status().isNotFound());

		mockMvc.perform(post("/api/v1/admin/jobs/missing-job/run").header("Authorization", bearerToken()))
				.andExpect(status().isNotFound());
	}

	private static String bearerToken() {
		return "Bearer " + ADMIN_TOKEN;
	}
}
