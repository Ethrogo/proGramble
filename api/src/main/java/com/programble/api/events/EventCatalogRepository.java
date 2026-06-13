package com.programble.api.events;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import com.programble.api.events.EventCatalogService.ProgrambleEvent;
import com.programble.api.events.EventCatalogService.SportDescriptor;

public interface EventCatalogRepository {

	Optional<SportDescriptor> findSport(String sportKey);

	List<ProgrambleEvent> findEventsForSportOnDate(long sportId, LocalDate date);

	Optional<ProgrambleEvent> findEvent(long eventId);
}
