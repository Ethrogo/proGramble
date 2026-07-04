package com.programble.api.web;

import java.time.Instant;
import java.util.Map;

import com.programble.api.config.ApiProperties;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("${programble.api.base-path}")
public class ApiRootController {

	private final ApiProperties apiProperties;

	public ApiRootController(ApiProperties apiProperties) {
		this.apiProperties = apiProperties;
	}

	@GetMapping
	public ResponseEntity<Map<String, Object>> root() {
		return ResponseEntity.ok(Map.of(
				"service", "programble-api",
				"environment", apiProperties.environment(),
				"version", "v1",
				"timestamp", Instant.now().toString(),
				"links", Map.of(
						"self", apiProperties.api().basePath(),
						"health", "/actuator/health",
						"liveness", "/actuator/health/liveness",
						"readiness", "/actuator/health/readiness",
						"metrics", "/actuator/metrics",
						"info", "/actuator/info"
				)
		));
	}
}
