package com.programble.api.oddsrefresh;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneId;

final class MlbTime {

	static final ZoneId EASTERN_ZONE = ZoneId.of("America/New_York");

	private MlbTime() {
	}

	static LocalDate toEasternDate(OffsetDateTime timestamp) {
		return timestamp.atZoneSameInstant(EASTERN_ZONE).toLocalDate();
	}

	static OffsetDateTime easternStartOfDay(LocalDate date) {
		return date.atStartOfDay(EASTERN_ZONE).toOffsetDateTime();
	}
}
