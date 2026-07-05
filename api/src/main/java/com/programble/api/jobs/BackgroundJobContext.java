package com.programble.api.jobs;

import java.time.Instant;

public record BackgroundJobContext(
		Trigger trigger,
		Instant startedAt
) {

	public enum Trigger {
		MANUAL,
		SCHEDULED
	}
}
