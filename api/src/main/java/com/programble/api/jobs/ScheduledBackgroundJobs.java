package com.programble.api.jobs;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class ScheduledBackgroundJobs {

	private final BackgroundJobService backgroundJobService;
	private final com.programble.api.config.BackgroundJobsProperties properties;

	public ScheduledBackgroundJobs(
			BackgroundJobService backgroundJobService,
			com.programble.api.config.BackgroundJobsProperties properties
	) {
		this.backgroundJobService = backgroundJobService;
		this.properties = properties;
	}

	@Scheduled(
			cron = "${programble.jobs.refresh-sports-data.cron:-}",
			zone = "${programble.jobs.time-zone:UTC}"
	)
	public void refreshSportsData() {
		if (!this.properties.schedulerEnabled()) {
			return;
		}
		this.backgroundJobService.runScheduled(SportsDataRefreshJob.KEY);
	}

	@Scheduled(
			cron = "${programble.jobs.refresh-odds.cron:-}",
			zone = "${programble.jobs.time-zone:UTC}"
	)
	public void refreshOdds() {
		if (!this.properties.schedulerEnabled()) {
			return;
		}
		this.backgroundJobService.runScheduled(OddsRefreshJob.KEY);
	}

	@Scheduled(
			cron = "${programble.jobs.refresh-derived-site-data.cron:-}",
			zone = "${programble.jobs.time-zone:UTC}"
	)
	public void refreshDerivedSiteData() {
		if (!this.properties.schedulerEnabled()) {
			return;
		}
		this.backgroundJobService.runScheduled(DerivedSiteDataRefreshJob.KEY);
	}
}
