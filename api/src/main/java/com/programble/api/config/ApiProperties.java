package com.programble.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "programble")
public record ApiProperties(
		String environment,
		Api api,
		Database database
) {

	public record Api(
			String basePath
	) {
	}

	public record Database(
			String url,
			String username,
			String password
	) {
	}
}
