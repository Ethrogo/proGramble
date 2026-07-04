package com.programble.api.jobs;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class ScheduledBackgroundJobs {

	private final BackgroundJobService backgroundJobService;

	public ScheduledBackgroundJobs(BackgroundJobService backgroundJobService) {
		this.backgroundJobService = backgroundJobService;
	}

	@Scheduled(
			cron = "${programble.jobs.refresh-sports-data.cron:-}",
			zone = "${programble.jobs.time-zone:UTC}"
	)
	public void refreshSportsData() {
		this.backgroundJobService.runScheduled(SportsDataRefreshJob.KEY);
	}

	@Scheduled(
			cron = "${programble.jobs.refresh-odds.cron:-}",
			zone = "${programble.jobs.time-zone:UTC}"
	)
	public void refreshOdds() {
		this.backgroundJobService.runScheduled(OddsRefreshJob.KEY);
	}

	@Scheduled(
			cron = "${programble.jobs.refresh-derived-site-data.cron:-}",
			zone = "${programble.jobs.time-zone:UTC}"
	)
	public void refreshDerivedSiteData() {
		this.backgroundJobService.runScheduled(DerivedSiteDataRefreshJob.KEY);
	}
}
