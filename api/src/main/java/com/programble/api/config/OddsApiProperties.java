package com.programble.api.config;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "programble.odds")
public record OddsApiProperties(
		String apiKey,
		String baseUrl,
		String mlbSportKey,
		String mlbPitcherStrikeoutsMarketKey,
		String eventDiscoveryMarketKey,
		List<String> bookmakers
) {

	public List<String> defaultBookmakers() {
		return this.bookmakers == null ? List.of() : this.bookmakers.stream()
				.map(String::trim)
				.filter(value -> !value.isEmpty())
				.toList();
	}
}
