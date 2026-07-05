package com.programble.api.jobs;

public class BackgroundJobAlreadyRunningException extends RuntimeException {

	public BackgroundJobAlreadyRunningException(String jobKey) {
		super("Background job is already running: " + jobKey);
	}
}
