package com.programble.api.jobs;

import java.util.Map;

import com.programble.api.oddsrefresh.MlbSportsDataRefreshService;
import org.springframework.stereotype.Component;

@Component
public class SportsDataRefreshJob implements BackgroundJob {

	public static final String KEY = "refresh-sports-data";
	private final MlbSportsDataRefreshService refreshService;

	public SportsDataRefreshJob(MlbSportsDataRefreshService refreshService) {
		this.refreshService = refreshService;
	}

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
		return "Refreshes the MLB schedule-backed event catalog and backfills historical tracked pitcher strikeout offers from the repo.";
	}

	@Override
	public BackgroundJobResult run(BackgroundJobContext context) {
		BackgroundJobResult result = this.refreshService.refresh();
		Map<String, Object> details = new java.util.LinkedHashMap<>(result.details());
		details.put("trigger", context.trigger().name());
		return new BackgroundJobResult(result.summary(), Map.copyOf(details));
	}
}
