package com.programble.api.jobs;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Component;

@Component
public class BackgroundJobRegistry {

	private final Map<String, BackgroundJob> jobsByKey;

	public BackgroundJobRegistry(List<BackgroundJob> jobs) {
		Map<String, BackgroundJob> indexedJobs = new LinkedHashMap<>();
		for (BackgroundJob job : jobs) {
			indexedJobs.put(job.key(), job);
		}
		this.jobsByKey = Map.copyOf(indexedJobs);
	}

	public List<BackgroundJob> jobs() {
		return this.jobsByKey.values().stream().toList();
	}

	public Optional<BackgroundJob> find(String jobKey) {
		return Optional.ofNullable(this.jobsByKey.get(jobKey));
	}
}
