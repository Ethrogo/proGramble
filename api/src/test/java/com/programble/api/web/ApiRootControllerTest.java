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
class ApiRootControllerTest {

	@Autowired
	private MockMvc mockMvc;

	@Test
	void apiRootExposesVersionedNamespace() throws Exception {
		mockMvc.perform(get("/api/v1"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.service").value("programble-api"))
				.andExpect(jsonPath("$.environment").value("test"))
				.andExpect(jsonPath("$.version").value("v1"))
				.andExpect(jsonPath("$.links.self").value("/api/v1"))
				.andExpect(jsonPath("$.links.health").value("/actuator/health"));
	}

	@Test
	void actuatorHealthEndpointIsAvailable() throws Exception {
		mockMvc.perform(get("/actuator/health"))
				.andExpect(status().isOk())
				.andExpect(jsonPath("$.status").value("UP"));
	}
}
