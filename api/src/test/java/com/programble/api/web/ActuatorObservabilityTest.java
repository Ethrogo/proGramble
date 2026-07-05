package com.programble.api.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
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
class ActuatorObservabilityTest {

	@Autowired
	private MockMvc mockMvc;

	@Test
	void livenessAndReadinessProbesAreAvailable() throws Exception {
		mockMvc.perform(get("/actuator/health/liveness"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.status").value("UP"));

		mockMvc.perform(get("/actuator/health/readiness"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.status").value("UP"));
	}

	@Test
	void metricsEndpointExposesHttpServerRequests() throws Exception {
		mockMvc.perform(get("/api/v1"))
				.andExpect(status().isOk());

		mockMvc.perform(get("/actuator/metrics/http.server.requests"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.name").value("http.server.requests"));
	}
}
