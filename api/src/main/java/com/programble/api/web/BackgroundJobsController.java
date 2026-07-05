package com.programble.api.web;

import java.util.List;

import com.programble.api.jobs.BackgroundJobAlreadyRunningException;
import com.programble.api.jobs.BackgroundJobNotFoundException;
import com.programble.api.jobs.BackgroundJobService;
import com.programble.api.jobs.BackgroundJobService.BackgroundJobRunResult;
import com.programble.api.jobs.BackgroundJobService.BackgroundJobSnapshot;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@RestController
@RequestMapping("${programble.api.base-path}/admin/jobs")
public class BackgroundJobsController {

	private final BackgroundJobService backgroundJobService;

	public BackgroundJobsController(BackgroundJobService backgroundJobService) {
		this.backgroundJobService = backgroundJobService;
	}

	@GetMapping
	public BackgroundJobsResponse jobs() {
		List<BackgroundJobSnapshot> jobs = this.backgroundJobService.listJobs();
		return new BackgroundJobsResponse(jobs.size(), jobs);
	}

	@GetMapping("/{jobKey}")
	public BackgroundJobSnapshot job(@PathVariable String jobKey) {
		try {
			return this.backgroundJobService.findJob(jobKey)
					.orElseThrow(() -> new BackgroundJobNotFoundException(jobKey));
		}
		catch (BackgroundJobNotFoundException exception) {
			throw new ResponseStatusException(HttpStatus.NOT_FOUND, exception.getMessage(), exception);
		}
	}

	@PostMapping("/{jobKey}/run")
	public BackgroundJobRunResult run(@PathVariable String jobKey) {
		try {
			return this.backgroundJobService.runManual(jobKey);
		}
		catch (BackgroundJobNotFoundException exception) {
			throw new ResponseStatusException(HttpStatus.NOT_FOUND, exception.getMessage(), exception);
		}
		catch (BackgroundJobAlreadyRunningException exception) {
			throw new ResponseStatusException(HttpStatus.CONFLICT, exception.getMessage(), exception);
		}
	}
}

record BackgroundJobsResponse(
		int count,
		List<BackgroundJobSnapshot> jobs
) {
}
