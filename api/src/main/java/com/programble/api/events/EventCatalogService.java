package com.programble.api.events;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import org.springframework.stereotype.Service;

@Service
public class EventCatalogService {

	private final List<ProgrambleEvent> events;
	private final Map<Long, ProgrambleEvent> eventsById;
	private final Map<String, SportDescriptor> sportsByKey;

	public EventCatalogService() {
		this.events = sampleEvents();
		this.eventsById = this.events.stream()
				.collect(LinkedHashMap::new, (map, event) -> map.put(event.id(), event), Map::putAll);
		this.sportsByKey = buildSportsByKey(this.events);
	}

	public Optional<SportDescriptor> findSport(String sportKey) {
		return Optional.ofNullable(this.sportsByKey.get(normalize(sportKey)));
	}

	public List<ProgrambleEvent> findEventsForSportOnDate(SportDescriptor sport, LocalDate date) {
		return this.events.stream()
				.filter(event -> event.sport().id() == sport.id())
				.filter(event -> event.scheduledStart().toLocalDate().equals(date))
				.sorted((left, right) -> left.scheduledStart().compareTo(right.scheduledStart()))
				.toList();
	}

	public Optional<ProgrambleEvent> findEvent(long eventId) {
		return Optional.ofNullable(this.eventsById.get(eventId));
	}

	private static Map<String, SportDescriptor> buildSportsByKey(List<ProgrambleEvent> events) {
		Map<String, SportDescriptor> sports = new LinkedHashMap<>();
		for (ProgrambleEvent event : events) {
			SportDescriptor sport = event.sport();
			sports.put(normalize(sport.code()), sport);
			sports.put(normalize(sport.slug()), sport);
		}
		return Map.copyOf(sports);
	}

	private static String normalize(String value) {
		return value == null ? "" : value.trim().toLowerCase(Locale.US);
	}

	private static List<ProgrambleEvent> sampleEvents() {
		SportDescriptor mlb = new SportDescriptor(1L, "MLB", "mlb", "Major League Baseball");
		SportDescriptor nba = new SportDescriptor(2L, "NBA", "nba", "National Basketball Association");
		SportDescriptor tennis = new SportDescriptor(3L, "TENNIS", "tennis", "Tennis");
		SportDescriptor golf = new SportDescriptor(4L, "GOLF", "golf", "Golf");

		CompetitionDescriptor mlbCompetition = new CompetitionDescriptor(11L, "MLB", "mlb", "MLB Regular Season", "TEAM");
		CompetitionDescriptor nbaCompetition = new CompetitionDescriptor(21L, "NBA", "nba", "NBA Finals", "TEAM");
		CompetitionDescriptor atpCompetition = new CompetitionDescriptor(31L, "ATP-500", "atp-500", "ATP 500", "INDIVIDUAL");
		CompetitionDescriptor pgaCompetition = new CompetitionDescriptor(41L, "PGA", "pga", "PGA Tour", "INDIVIDUAL");

		return List.of(
				new ProgrambleEvent(
						1001L,
						mlb,
						mlbCompetition,
						"yankees-red-sox-2026-06-13",
						"mlb-20260613-nyy-bos",
						"GAME",
						"SCHEDULED",
						"2026",
						"Regular Season",
						OffsetDateTime.parse("2026-06-13T19:10:00-04:00"),
						true,
						new VenueDescriptor("Fenway Park", "Boston", "US"),
						List.of(
								new EventParticipantDescriptor(501L, "TEAM", "AWAY", "New York Yankees", "NYY", null, 1, false, true),
								new EventParticipantDescriptor(502L, "TEAM", "HOME", "Boston Red Sox", "BOS", null, 2, true, false)
						)
				),
				new ProgrambleEvent(
						1002L,
						mlb,
						mlbCompetition,
						"dodgers-giants-2026-06-14",
						"mlb-20260614-lad-sf",
						"GAME",
						"SCHEDULED",
						"2026",
						"Regular Season",
						OffsetDateTime.parse("2026-06-14T20:10:00-07:00"),
						true,
						new VenueDescriptor("Oracle Park", "San Francisco", "US"),
						List.of(
								new EventParticipantDescriptor(503L, "TEAM", "AWAY", "Los Angeles Dodgers", "LAD", null, 1, false, true),
								new EventParticipantDescriptor(504L, "TEAM", "HOME", "San Francisco Giants", "SF", null, 2, true, false)
						)
				),
				new ProgrambleEvent(
						2001L,
						nba,
						nbaCompetition,
						"knicks-celtics-2026-06-13",
						"nba-20260613-nyk-bos",
						"GAME",
						"SCHEDULED",
						"2025-26",
						"Finals Game 4",
						OffsetDateTime.parse("2026-06-13T20:30:00-04:00"),
						true,
						new VenueDescriptor("TD Garden", "Boston", "US"),
						List.of(
								new EventParticipantDescriptor(601L, "TEAM", "AWAY", "New York Knicks", "NYK", null, 1, false, true),
								new EventParticipantDescriptor(602L, "TEAM", "HOME", "Boston Celtics", "BOS", null, 2, true, false)
						)
				),
				new ProgrambleEvent(
						3001L,
						tennis,
						atpCompetition,
						"alcaraz-sinner-2026-06-13",
						"atp-20260613-car-sin",
						"MATCH",
						"SCHEDULED",
						"2026",
						"Quarterfinal",
						OffsetDateTime.parse("2026-06-13T14:00:00+01:00"),
						true,
						new VenueDescriptor("The Queen's Club", "London", "GB"),
						List.of(
								new EventParticipantDescriptor(701L, "PLAYER", "PLAYER_ONE", "Carlos Alcaraz", "Alcaraz", 1, 1, null, null),
								new EventParticipantDescriptor(702L, "PLAYER", "PLAYER_TWO", "Jannik Sinner", "Sinner", 2, 2, null, null)
						)
				),
				new ProgrambleEvent(
						4001L,
						golf,
						pgaCompetition,
						"us-open-2026",
						"pga-2026-us-open",
						"TOURNAMENT",
						"SCHEDULED",
						"2026",
						"Week 24",
						OffsetDateTime.parse("2026-06-13T08:00:00-04:00"),
						false,
						new VenueDescriptor("Oakmont Country Club", "Oakmont", "US"),
						List.of(
								new EventParticipantDescriptor(801L, "PLAYER", "FIELD", "Scottie Scheffler", "Scheffler", 1, 1, null, null),
								new EventParticipantDescriptor(802L, "PLAYER", "FIELD", "Rory McIlroy", "McIlroy", 2, 2, null, null)
						)
				)
		);
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
