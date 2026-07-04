package com.programble.api.jobs;

import java.util.Map;

import org.springframework.stereotype.Component;

@Component
public class DerivedSiteDataRefreshJob implements BackgroundJob {

	public static final String KEY = "refresh-derived-site-data";

	@Override
	public String key() {
		return KEY;
	}

	@Override
	public String displayName() {
		return "Refresh derived site data";
	}

	@Override
	public String description() {
		return "Placeholder hook for computed site-facing data such as slates, featured cards, and cached summaries.";
	}

	@Override
	public BackgroundJobResult run(BackgroundJobContext context) {
		return new BackgroundJobResult(
				"Derived site data refresh placeholder completed",
				Map.of(
						"placeholder", true,
						"artifactsTouched", 0,
						"trigger", context.trigger().name()
				)
		);
	}
}
