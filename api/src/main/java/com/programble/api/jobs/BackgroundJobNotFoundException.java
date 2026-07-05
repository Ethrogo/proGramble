package com.programble.api.jobs;

public class BackgroundJobNotFoundException extends RuntimeException {

	public BackgroundJobNotFoundException(String jobKey) {
		super("Unknown background job: " + jobKey);
	}
}
