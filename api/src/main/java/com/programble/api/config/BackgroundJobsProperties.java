package com.programble.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.util.StringUtils;

@ConfigurationProperties(prefix = "programble.jobs")
public record BackgroundJobsProperties(
		String timeZone,
		JobSchedule refreshSportsData,
		JobSchedule refreshOdds,
		JobSchedule refreshDerivedSiteData
) {

	public BackgroundJobsProperties {
		timeZone = StringUtils.hasText(timeZone) ? timeZone : "UTC";
		refreshSportsData = refreshSportsData == null ? new JobSchedule(Scheduled.CRON_DISABLED) : refreshSportsData;
		refreshOdds = refreshOdds == null ? new JobSchedule(Scheduled.CRON_DISABLED) : refreshOdds;
		refreshDerivedSiteData = refreshDerivedSiteData == null ? new JobSchedule(Scheduled.CRON_DISABLED) : refreshDerivedSiteData;
	}

	public JobSchedule scheduleFor(String jobKey) {
		return switch (jobKey) {
			case "refresh-sports-data" -> this.refreshSportsData;
			case "refresh-odds" -> this.refreshOdds;
			case "refresh-derived-site-data" -> this.refreshDerivedSiteData;
			default -> new JobSchedule(Scheduled.CRON_DISABLED);
		};
	}

	public record JobSchedule(
			String cron
	) {

		public JobSchedule {
			cron = StringUtils.hasText(cron) ? cron : Scheduled.CRON_DISABLED;
		}

		public boolean scheduleEnabled() {
			return !Scheduled.CRON_DISABLED.equals(this.cron);
		}
	}
}
