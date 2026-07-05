package com.programble.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "programble.mlb")
public record MlbStatsApiProperties(
		String statsApiBaseUrl
) {
}
