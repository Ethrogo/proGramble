package com.programble.api.web;

import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.List;

import com.programble.api.events.EventCatalogService;
import com.programble.api.events.EventCatalogService.CompetitionDescriptor;
import com.programble.api.events.EventCatalogService.EventParticipantDescriptor;
import com.programble.api.events.EventCatalogService.ProgrambleEvent;
import com.programble.api.events.EventCatalogService.SportDescriptor;
import com.programble.api.events.EventCatalogService.VenueDescriptor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import static org.springframework.http.HttpStatus.NOT_FOUND;

@RestController
@RequestMapping("${programble.api.base-path}")
public class SportEventsController {

	private final EventCatalogService eventCatalogService;

	public SportEventsController(EventCatalogService eventCatalogService) {
		this.eventCatalogService = eventCatalogService;
	}

	@GetMapping("/sports/{sport}/events")
	public EventListResponse sportEvents(
			@PathVariable String sport,
			@RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date
	) {
		SportDescriptor sportDescriptor = this.eventCatalogService.findSport(sport)
				.orElseThrow(() -> new ResponseStatusException(NOT_FOUND, "Unknown sport: " + sport));

		LocalDate requestedDate = date == null ? LocalDate.now(ZoneOffset.UTC) : date;
		List<ProgrambleEvent> events = this.eventCatalogService.findEventsForSportOnDate(sportDescriptor, requestedDate);

		return new EventListResponse(
				toSportResponse(sportDescriptor),
				requestedDate,
				events.size(),
				events.stream().map(SportEventsController::toEventSummaryResponse).toList()
		);
	}

	@GetMapping("/events/{eventId}")
	public EventDetailResponse eventDetail(@PathVariable long eventId) {
		ProgrambleEvent event = this.eventCatalogService.findEvent(eventId)
				.orElseThrow(() -> new ResponseStatusException(NOT_FOUND, "Unknown event id: " + eventId));

		return new EventDetailResponse(
				event.id(),
				toSportResponse(event.sport()),
				toCompetitionResponse(event.competition()),
				event.slug(),
				event.externalRef(),
				event.eventType(),
				event.status(),
				event.seasonLabel(),
				event.roundLabel(),
				event.scheduledStart(),
				event.startTimeConfirmed(),
				toVenueResponse(event.venue()),
				event.participants().stream().map(SportEventsController::toParticipantResponse).toList()
		);
	}

	private static EventSummaryResponse toEventSummaryResponse(ProgrambleEvent event) {
		return new EventSummaryResponse(
				event.id(),
				event.slug(),
				event.eventType(),
				event.status(),
				event.seasonLabel(),
				event.roundLabel(),
				event.scheduledStart(),
				event.startTimeConfirmed(),
				toCompetitionResponse(event.competition()),
				toVenueResponse(event.venue()),
				event.participants().stream().map(SportEventsController::toParticipantResponse).toList()
		);
	}

	private static SportResponse toSportResponse(SportDescriptor sport) {
		return new SportResponse(sport.code(), sport.slug(), sport.name());
	}

	private static CompetitionResponse toCompetitionResponse(CompetitionDescriptor competition) {
		return new CompetitionResponse(
				competition.code(),
				competition.slug(),
				competition.name(),
				competition.competitionType()
		);
	}

	private static VenueResponse toVenueResponse(VenueDescriptor venue) {
		return new VenueResponse(venue.name(), venue.city(), venue.countryCode());
	}

	private static EventParticipantResponse toParticipantResponse(EventParticipantDescriptor participant) {
		return new EventParticipantResponse(
				participant.id(),
				participant.type(),
				participant.roleCode(),
				participant.displayName(),
				participant.shortName(),
				participant.seedValue(),
				participant.sortOrder(),
				participant.isHome(),
				participant.isAway()
		);
	}
}

record EventListResponse(
		SportResponse sport,
		LocalDate date,
		int count,
		List<EventSummaryResponse> events
) {
}

record EventDetailResponse(
		long id,
		SportResponse sport,
		CompetitionResponse competition,
		String slug,
		String externalRef,
		String eventType,
		String status,
		String seasonLabel,
		String roundLabel,
		java.time.OffsetDateTime scheduledStart,
		boolean startTimeConfirmed,
		VenueResponse venue,
		List<EventParticipantResponse> participants
) {
}

record EventSummaryResponse(
		long id,
		String slug,
		String eventType,
		String status,
		String seasonLabel,
		String roundLabel,
		java.time.OffsetDateTime scheduledStart,
		boolean startTimeConfirmed,
		CompetitionResponse competition,
		VenueResponse venue,
		List<EventParticipantResponse> participants
) {
}

record SportResponse(
		String code,
		String slug,
		String name
) {
}

record CompetitionResponse(
		String code,
		String slug,
		String name,
		String competitionType
) {
}

record VenueResponse(
		String name,
		String city,
		String countryCode
) {
}

record EventParticipantResponse(
		long id,
		String type,
		String roleCode,
		String displayName,
		String shortName,
		Integer seedValue,
		Integer sortOrder,
		Boolean isHome,
		Boolean isAway
) {
}
