package com.programble.api.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class BackgroundJobsControllerTest {

	@Autowired
	private MockMvc mockMvc;

	@Test
	void listsJobsAndRunsPlaceholderManually() throws Exception {
		mockMvc.perform(get("/api/v1/admin/jobs"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.count").value(3))
				.andExpect(jsonPath("$.jobs[0].key").value("refresh-derived-site-data"))
				.andExpect(jsonPath("$.jobs[0].scheduleEnabled").value(false))
				.andExpect(jsonPath("$.jobs[1].key").value("refresh-odds"))
				.andExpect(jsonPath("$.jobs[2].key").value("refresh-sports-data"));

		mockMvc.perform(post("/api/v1/admin/jobs/refresh-odds/run"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.trigger").value("MANUAL"))
				.andExpect(jsonPath("$.summary").value("Odds refresh placeholder completed"))
				.andExpect(jsonPath("$.details.placeholder").value(true))
				.andExpect(jsonPath("$.job.key").value("refresh-odds"))
				.andExpect(jsonPath("$.job.totalRuns").value(1))
				.andExpect(jsonPath("$.job.successfulRuns").value(1))
				.andExpect(jsonPath("$.job.failedRuns").value(0))
				.andExpect(jsonPath("$.job.lastTrigger").value("MANUAL"));

		mockMvc.perform(get("/api/v1/admin/jobs/refresh-odds"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.key").value("refresh-odds"))
				.andExpect(jsonPath("$.scheduleEnabled").value(false))
				.andExpect(jsonPath("$.totalRuns").value(1))
				.andExpect(jsonPath("$.successfulRuns").value(1))
				.andExpect(jsonPath("$.lastSummary").value("Odds refresh placeholder completed"))
				.andExpect(jsonPath("$.lastDetails.placeholder").value(true));
	}

	@Test
	void returnsNotFoundForUnknownJob() throws Exception {
		mockMvc.perform(get("/api/v1/admin/jobs/missing-job"))
				.andExpect(status().isNotFound());

		mockMvc.perform(post("/api/v1/admin/jobs/missing-job/run"))
				.andExpect(status().isNotFound());
	}
}
