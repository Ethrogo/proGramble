package com.programble.api.jobs;

public interface BackgroundJob {

	String key();

	String displayName();

	String description();

	BackgroundJobResult run(BackgroundJobContext context);
}
