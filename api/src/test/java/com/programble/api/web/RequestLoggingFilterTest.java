package com.programble.api.web;

import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.blankOrNullString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
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
class RequestLoggingFilterTest {

	@Autowired
	private MockMvc mockMvc;

	@Test
	void applicationRequestsReceiveGeneratedRequestId() throws Exception {
		mockMvc.perform(get("/api/v1"))
				.andExpect(status().isOk())
				.andExpect(header().string("X-Request-Id", not(blankOrNullString())));
	}

	@Test
	void applicationRequestsPreserveCallerRequestId() throws Exception {
		mockMvc.perform(get("/api/v1").header("X-Request-Id", "req-12345"))
				.andExpect(status().isOk())
				.andExpect(header().string("X-Request-Id", "req-12345"));
	}

	@Test
	void actuatorProbeRequestsSkipRequestIdDecoration() throws Exception {
		mockMvc.perform(get("/actuator/health"))
				.andExpect(status().isOk())
				.andExpect(header().doesNotExist("X-Request-Id"));
	}
}
