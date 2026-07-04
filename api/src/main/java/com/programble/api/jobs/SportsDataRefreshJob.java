package com.programble.api.jobs;

import java.util.Map;

import org.springframework.stereotype.Component;

@Component
public class SportsDataRefreshJob implements BackgroundJob {

	public static final String KEY = "refresh-sports-data";

	@Override
	public String key() {
		return KEY;
	}

	@Override
	public String displayName() {
		return "Refresh sports data";
	}

	@Override
	public String description() {
		return "Placeholder ingestion hook for competitions, teams, players, and event catalog data.";
	}

	@Override
	public BackgroundJobResult run(BackgroundJobContext context) {
		return new BackgroundJobResult(
				"Sports data refresh placeholder completed",
				Map.of(
						"placeholder", true,
						"datasetsTouched", 0,
						"trigger", context.trigger().name()
				)
		);
	}
}
