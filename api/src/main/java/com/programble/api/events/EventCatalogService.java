package com.programble.api.events;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

@Service
public class EventCatalogService {

	private final EventCatalogRepository eventCatalogRepository;

	public EventCatalogService(EventCatalogRepository eventCatalogRepository) {
		this.eventCatalogRepository = eventCatalogRepository;
	}

	public Optional<SportDescriptor> findSport(String sportKey) {
		return this.eventCatalogRepository.findSport(sportKey);
	}

	public List<ProgrambleEvent> findEventsForSportOnDate(SportDescriptor sport, LocalDate date) {
		return this.eventCatalogRepository.findEventsForSportOnDate(sport.id(), date);
	}

	public Optional<ProgrambleEvent> findEvent(long eventId) {
		return this.eventCatalogRepository.findEvent(eventId);
	}

	public record SportDescriptor(
			long id,
			String code,
			String slug,
			String name
	) {
	}

	public record CompetitionDescriptor(
			long id,
			String code,
			String slug,
			String name,
			String competitionType
	) {
	}

	public record VenueDescriptor(
			String name,
			String city,
			String countryCode
	) {
	}

	public record EventParticipantDescriptor(
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

	public record ProgrambleEvent(
			long id,
			SportDescriptor sport,
			CompetitionDescriptor competition,
			String slug,
			String externalRef,
			String eventType,
			String status,
			String seasonLabel,
			String roundLabel,
			OffsetDateTime scheduledStart,
			boolean startTimeConfirmed,
			VenueDescriptor venue,
			List<EventParticipantDescriptor> participants
	) {
	}
}
