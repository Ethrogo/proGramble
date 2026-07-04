package com.programble.api.jobs;

import java.util.Map;

import org.springframework.stereotype.Component;

@Component
public class OddsRefreshJob implements BackgroundJob {

	public static final String KEY = "refresh-odds";

	@Override
	public String key() {
		return KEY;
	}

	@Override
	public String displayName() {
		return "Refresh sportsbook odds";
	}

	@Override
	public String description() {
		return "Placeholder ingestion hook for sportsbook lines, prices, and market availability.";
	}

	@Override
	public BackgroundJobResult run(BackgroundJobContext context) {
		return new BackgroundJobResult(
				"Odds refresh placeholder completed",
				Map.of(
						"placeholder", true,
						"offersTouched", 0,
						"trigger", context.trigger().name()
				)
		);
	}
}
