package com.programble.api.jobs;

import java.util.Map;

import com.programble.api.oddsrefresh.MlbPitcherStrikeoutsRefreshService;
import org.springframework.stereotype.Component;

@Component
public class OddsRefreshJob implements BackgroundJob {

	public static final String KEY = "refresh-odds";
	private final MlbPitcherStrikeoutsRefreshService refreshService;

	public OddsRefreshJob(MlbPitcherStrikeoutsRefreshService refreshService) {
		this.refreshService = refreshService;
	}

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
		return "Refreshes MLB pitcher strikeout offers from live Odds API data when configured, or falls back to tracked daily-card artifacts when live odds are unavailable.";
	}

	@Override
	public BackgroundJobResult run(BackgroundJobContext context) {
		BackgroundJobResult result = this.refreshService.refresh();
		return new BackgroundJobResult(
				result.summary(),
				mergeDetails(result.details(), context.trigger().name())
		);
	}

	private static Map<String, Object> mergeDetails(Map<String, Object> details, String trigger) {
		Map<String, Object> merged = new java.util.LinkedHashMap<>(details);
		merged.put("trigger", trigger);
		return Map.copyOf(merged);
	}
}
