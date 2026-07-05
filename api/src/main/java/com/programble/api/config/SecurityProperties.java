package com.programble.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "programble.security")
public record SecurityProperties(
		String adminApiToken
) {
}
