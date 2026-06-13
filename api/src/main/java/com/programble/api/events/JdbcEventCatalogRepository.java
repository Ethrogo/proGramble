package com.programble.api.events;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import com.programble.api.events.EventCatalogService.CompetitionDescriptor;
import com.programble.api.events.EventCatalogService.EventParticipantDescriptor;
import com.programble.api.events.EventCatalogService.ProgrambleEvent;
import com.programble.api.events.EventCatalogService.SportDescriptor;
import com.programble.api.events.EventCatalogService.VenueDescriptor;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class JdbcEventCatalogRepository implements EventCatalogRepository {

	private static final RowMapper<SportDescriptor> SPORT_ROW_MAPPER = (resultSet, rowNum) ->
			new SportDescriptor(
					resultSet.getLong("id"),
					resultSet.getString("code"),
					resultSet.getString("slug"),
					resultSet.getString("name")
			);

	private final NamedParameterJdbcTemplate jdbcTemplate;

	public JdbcEventCatalogRepository(NamedParameterJdbcTemplate jdbcTemplate) {
		this.jdbcTemplate = jdbcTemplate;
	}

	@Override
	public Optional<SportDescriptor> findSport(String sportKey) {
		String normalizedSportKey = normalize(sportKey);
		if (normalizedSportKey.isEmpty()) {
			return Optional.empty();
		}

		List<SportDescriptor> sports = this.jdbcTemplate.query(
				"""
				select id, code, slug, name
				from sports
				where lower(code) = :sportKey
				   or lower(slug) = :sportKey
				limit 1
				""",
				new MapSqlParameterSource("sportKey", normalizedSportKey),
				SPORT_ROW_MAPPER
		);

		return sports.stream().findFirst();
	}

	@Override
	public List<ProgrambleEvent> findEventsForSportOnDate(long sportId, LocalDate date) {
		OffsetDateTime start = date.atStartOfDay().atOffset(ZoneOffset.UTC);
		OffsetDateTime end = start.plusDays(1);

		List<EventBaseRow> eventRows = this.jdbcTemplate.query(
				"""
				select
				    e.id,
				    e.slug,
				    e.external_ref,
				    e.event_type,
				    e.status,
				    e.season_label,
				    e.round_label,
				    e.scheduled_start,
				    e.start_time_confirmed,
				    e.venue_name,
				    e.venue_city,
				    e.venue_country_code,
				    s.id as sport_id,
				    s.code as sport_code,
				    s.slug as sport_slug,
				    s.name as sport_name,
				    c.id as competition_id,
				    c.code as competition_code,
				    c.slug as competition_slug,
				    c.name as competition_name,
				    c.competition_type
				from events e
				join sports s on s.id = e.sport_id
				join competitions c on c.id = e.competition_id
				where e.sport_id = :sportId
				  and e.scheduled_start >= :start
				  and e.scheduled_start < :end
				order by e.scheduled_start, e.id
				""",
				new MapSqlParameterSource()
						.addValue("sportId", sportId)
						.addValue("start", start)
						.addValue("end", end),
				(resultSet, rowNum) -> mapEventBase(resultSet)
		);

		return hydrateEvents(eventRows);
	}

	@Override
	public Optional<ProgrambleEvent> findEvent(long eventId) {
		List<EventBaseRow> eventRows = this.jdbcTemplate.query(
				"""
				select
				    e.id,
				    e.slug,
				    e.external_ref,
				    e.event_type,
				    e.status,
				    e.season_label,
				    e.round_label,
				    e.scheduled_start,
				    e.start_time_confirmed,
				    e.venue_name,
				    e.venue_city,
				    e.venue_country_code,
				    s.id as sport_id,
				    s.code as sport_code,
				    s.slug as sport_slug,
				    s.name as sport_name,
				    c.id as competition_id,
				    c.code as competition_code,
				    c.slug as competition_slug,
				    c.name as competition_name,
				    c.competition_type
				from events e
				join sports s on s.id = e.sport_id
				join competitions c on c.id = e.competition_id
				where e.id = :eventId
				""",
				new MapSqlParameterSource("eventId", eventId),
				(resultSet, rowNum) -> mapEventBase(resultSet)
		);

		return hydrateEvents(eventRows).stream().findFirst();
	}

	private List<ProgrambleEvent> hydrateEvents(List<EventBaseRow> eventRows) {
		if (eventRows.isEmpty()) {
			return List.of();
		}

		Map<Long, List<EventParticipantDescriptor>> participantsByEventId = loadParticipants(
				eventRows.stream().map(EventBaseRow::id).toList()
		);

		return eventRows.stream()
				.map(row -> new ProgrambleEvent(
						row.id(),
						row.sport(),
						row.competition(),
						row.slug(),
						row.externalRef(),
						row.eventType(),
						row.status(),
						row.seasonLabel(),
						row.roundLabel(),
						row.scheduledStart(),
						row.startTimeConfirmed(),
						row.venue(),
						participantsByEventId.getOrDefault(row.id(), List.of())
				))
				.toList();
	}

	private Map<Long, List<EventParticipantDescriptor>> loadParticipants(List<Long> eventIds) {
		List<EventParticipantRow> participantRows = this.jdbcTemplate.query(
				"""
				select
				    ep.id,
				    ep.event_id,
				    case
				        when ep.team_id is not null then 'TEAM'
				        else 'PLAYER'
				    end as participant_type,
				    ep.role_code,
				    coalesce(t.full_name, p.display_name) as display_name,
				    coalesce(t.short_name, p.display_name) as short_name,
				    ep.seed_value,
				    ep.sort_order,
				    ep.is_home,
				    ep.is_away
				from event_participants ep
				left join teams t on t.id = ep.team_id
				left join players p on p.id = ep.player_id
				where ep.event_id in (:eventIds)
				order by ep.event_id, ep.sort_order nulls last, ep.id
				""",
				new MapSqlParameterSource("eventIds", eventIds),
				(resultSet, rowNum) -> new EventParticipantRow(
						resultSet.getLong("id"),
						resultSet.getLong("event_id"),
						new EventParticipantDescriptor(
								resultSet.getLong("id"),
								resultSet.getString("participant_type"),
								resultSet.getString("role_code"),
								resultSet.getString("display_name"),
								resultSet.getString("short_name"),
								(Integer) resultSet.getObject("seed_value"),
								(Integer) resultSet.getObject("sort_order"),
								(Boolean) resultSet.getObject("is_home"),
								(Boolean) resultSet.getObject("is_away")
						)
				)
		);

		Map<Long, List<EventParticipantDescriptor>> participantsByEventId = new LinkedHashMap<>();
		for (EventParticipantRow row : participantRows) {
			participantsByEventId.computeIfAbsent(row.eventId(), ignored -> new ArrayList<>())
					.add(row.participant());
		}

		return participantsByEventId.entrySet().stream()
				.collect(Collectors.toMap(Map.Entry::getKey, entry -> List.copyOf(entry.getValue())));
	}

	private static EventBaseRow mapEventBase(ResultSet resultSet) throws SQLException {
		return new EventBaseRow(
				resultSet.getLong("id"),
				new SportDescriptor(
						resultSet.getLong("sport_id"),
						resultSet.getString("sport_code"),
						resultSet.getString("sport_slug"),
						resultSet.getString("sport_name")
				),
				new CompetitionDescriptor(
						resultSet.getLong("competition_id"),
						resultSet.getString("competition_code"),
						resultSet.getString("competition_slug"),
						resultSet.getString("competition_name"),
						resultSet.getString("competition_type")
				),
				resultSet.getString("slug"),
				resultSet.getString("external_ref"),
				resultSet.getString("event_type"),
				resultSet.getString("status"),
				resultSet.getString("season_label"),
				resultSet.getString("round_label"),
				resultSet.getObject("scheduled_start", OffsetDateTime.class),
				resultSet.getBoolean("start_time_confirmed"),
				new VenueDescriptor(
						resultSet.getString("venue_name"),
						resultSet.getString("venue_city"),
						resultSet.getString("venue_country_code")
				)
		);
	}

	private static String normalize(String value) {
		return value == null ? "" : value.trim().toLowerCase(Locale.US);
	}

	private record EventBaseRow(
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
			VenueDescriptor venue
	) {
	}

	private record EventParticipantRow(
			long id,
			long eventId,
			EventParticipantDescriptor participant
	) {
	}
}
