package com.programble.api.jobs;

import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.locks.ReentrantLock;

import com.programble.api.config.BackgroundJobsProperties;
import com.programble.api.jobs.BackgroundJobContext.Trigger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class BackgroundJobService {

	private static final Logger log = LoggerFactory.getLogger(BackgroundJobService.class);

	private final BackgroundJobRegistry registry;
	private final BackgroundJobsProperties properties;
	private final ConcurrentMap<String, MutableJobState> stateByJob = new ConcurrentHashMap<>();

	public BackgroundJobService(BackgroundJobRegistry registry, BackgroundJobsProperties properties) {
		this.registry = registry;
		this.properties = properties;
	}

	public List<BackgroundJobSnapshot> listJobs() {
		return this.registry.jobs().stream()
				.sorted(Comparator.comparing(BackgroundJob::key))
				.map(this::snapshot)
				.toList();
	}

	public Optional<BackgroundJobSnapshot> findJob(String jobKey) {
		return this.registry.find(jobKey).map(this::snapshot);
	}

	public BackgroundJobRunResult runManual(String jobKey) {
		BackgroundJob job = this.registry.find(jobKey)
				.orElseThrow(() -> new BackgroundJobNotFoundException(jobKey));
		MutableJobState state = this.stateByJob.computeIfAbsent(job.key(), ignored -> new MutableJobState());

		if (!state.lock.tryLock()) {
			throw new BackgroundJobAlreadyRunningException(job.key());
		}

		try {
			return run(job, state, Trigger.MANUAL);
		}
		finally {
			state.lock.unlock();
		}
	}

	public void runScheduled(String jobKey) {
		BackgroundJob job = this.registry.find(jobKey)
				.orElseThrow(() -> new BackgroundJobNotFoundException(jobKey));
		MutableJobState state = this.stateByJob.computeIfAbsent(job.key(), ignored -> new MutableJobState());

		if (!state.lock.tryLock()) {
			log.info("Skipping scheduled background job because a prior run is still active: {}", job.key());
			return;
		}

		try {
			run(job, state, Trigger.SCHEDULED);
		}
		catch (RuntimeException exception) {
			log.error("Scheduled background job failed: {}", job.key(), exception);
		}
		finally {
			state.lock.unlock();
		}
	}

	private BackgroundJobRunResult run(BackgroundJob job, MutableJobState state, Trigger trigger) {
		Instant startedAt = Instant.now();
		state.markStarted(trigger, startedAt);
		log.info("Starting background job {} with trigger {}", job.key(), trigger);

		try {
			BackgroundJobResult result = job.run(new BackgroundJobContext(trigger, startedAt));
			Instant finishedAt = Instant.now();
			state.markSuccess(finishedAt, result);
			log.info("Completed background job {} with trigger {}", job.key(), trigger);
			return new BackgroundJobRunResult(
					trigger,
					result.summary(),
					result.details(),
					snapshot(job, state)
			);
		}
		catch (RuntimeException exception) {
			Instant finishedAt = Instant.now();
			state.markFailure(finishedAt, exception);
			log.error("Background job failed {} with trigger {}", job.key(), trigger, exception);
			throw exception;
		}
	}

	private BackgroundJobSnapshot snapshot(BackgroundJob job) {
		MutableJobState state = this.stateByJob.computeIfAbsent(job.key(), ignored -> new MutableJobState());
		return snapshot(job, state);
	}

	private BackgroundJobSnapshot snapshot(BackgroundJob job, MutableJobState state) {
		BackgroundJobsProperties.JobSchedule schedule = this.properties.scheduleFor(job.key());
		return state.snapshot(
				job.key(),
				job.displayName(),
				job.description(),
				schedule.cron(),
				schedule.scheduleEnabled(),
				this.properties.timeZone()
		);
	}

	public record BackgroundJobRunResult(
			Trigger trigger,
			String summary,
			Map<String, Object> details,
			BackgroundJobSnapshot job
	) {
	}

	public record BackgroundJobSnapshot(
			String key,
			String displayName,
			String description,
			String cron,
			boolean scheduleEnabled,
			String timeZone,
			boolean running,
			long totalRuns,
			long successfulRuns,
			long failedRuns,
			Trigger lastTrigger,
			Instant lastStartedAt,
			Instant lastFinishedAt,
			Instant lastSucceededAt,
			Instant lastFailedAt,
			String lastSummary,
			String lastError,
			Map<String, Object> lastDetails
	) {
	}

	private static final class MutableJobState {

		private final ReentrantLock lock = new ReentrantLock();
		private boolean running;
		private long totalRuns;
		private long successfulRuns;
		private long failedRuns;
		private Trigger lastTrigger;
		private Instant lastStartedAt;
		private Instant lastFinishedAt;
		private Instant lastSucceededAt;
		private Instant lastFailedAt;
		private String lastSummary;
		private String lastError;
		private Map<String, Object> lastDetails = Map.of();

		synchronized void markStarted(Trigger trigger, Instant startedAt) {
			this.running = true;
			this.totalRuns++;
			this.lastTrigger = trigger;
			this.lastStartedAt = startedAt;
			this.lastFinishedAt = null;
			this.lastError = null;
		}

		synchronized void markSuccess(Instant finishedAt, BackgroundJobResult result) {
			this.running = false;
			this.successfulRuns++;
			this.lastFinishedAt = finishedAt;
			this.lastSucceededAt = finishedAt;
			this.lastSummary = result.summary();
			this.lastDetails = result.details();
			this.lastError = null;
		}

		synchronized void markFailure(Instant finishedAt, RuntimeException exception) {
			this.running = false;
			this.failedRuns++;
			this.lastFinishedAt = finishedAt;
			this.lastFailedAt = finishedAt;
			this.lastSummary = "FAILED";
			this.lastDetails = Map.of();
			this.lastError = exception.getMessage();
		}

		synchronized BackgroundJobSnapshot snapshot(
				String key,
				String displayName,
				String description,
				String cron,
				boolean scheduleEnabled,
				String timeZone
		) {
			return new BackgroundJobSnapshot(
					key,
					displayName,
					description,
					cron,
					scheduleEnabled,
					timeZone,
					this.running,
					this.totalRuns,
					this.successfulRuns,
					this.failedRuns,
					this.lastTrigger,
					this.lastStartedAt,
					this.lastFinishedAt,
					this.lastSucceededAt,
					this.lastFailedAt,
					this.lastSummary,
					this.lastError,
					this.lastDetails
			);
		}
	}
}
