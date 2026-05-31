package com.programble.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "programble")
public record ApiProperties(
		String environment,
		Api api
) {

	public record Api(
			String basePath
	) {
	}
}
