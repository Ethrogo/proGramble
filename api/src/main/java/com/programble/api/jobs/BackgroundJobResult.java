package com.programble.api.jobs;

import java.util.Map;

public record BackgroundJobResult(
		String summary,
		Map<String, Object> details
) {

	public BackgroundJobResult {
		details = details == null ? Map.of() : Map.copyOf(details);
	}
}
