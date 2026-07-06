package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.sql.PreparedStatement;
import java.text.Normalizer;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.stereotype.Repository;

@Repository
public class JdbcOddsRefreshRepository implements OddsRefreshRepository {

	private static final RowMapper<EventMatchCandidate> EVENT_MATCH_CANDIDATE_ROW_MAPPER = (resultSet, rowNum) ->
			new EventMatchCandidate(
					resultSet.getLong("event_id"),
					resultSet.getLong("sport_id"),
					resultSet.getObject("scheduled_start", OffsetDateTime.class),
					resultSet.getLong("home_team_id"),
					resultSet.getLong("away_team_id"),
					resultSet.getString("home_team_code"),
					resultSet.getString("away_team_code")
			);

	private final JdbcTemplate jdbcTemplate;
	private final NamedParameterJdbcTemplate namedParameterJdbcTemplate;

	public JdbcOddsRefreshRepository(JdbcTemplate jdbcTemplate, NamedParameterJdbcTemplate namedParameterJdbcTemplate) {
		this.jdbcTemplate = jdbcTemplate;
		this.namedParameterJdbcTemplate = namedParameterJdbcTemplate;
	}

	@Override
	public Optional<EventMatch> findMlbEvent(OffsetDateTime commenceTime, String homeTeam, String awayTeam) {
		OffsetDateTime windowStart = commenceTime.minusHours(12);
		OffsetDateTime windowEnd = commenceTime.plusHours(12);

		List<EventMatchCandidate> candidates = this.namedParameterJdbcTemplate.query(
				baseEventMatchSelect()
						+ """
						where lower(s.code) = 'mlb'
						  and e.scheduled_start >= :windowStart
						  and e.scheduled_start <= :windowEnd
						  and lower(home_t.full_name) = :homeTeam
						  and lower(away_t.full_name) = :awayTeam
						""",
				new MapSqlParameterSource()
						.addValue("windowStart", windowStart)
						.addValue("windowEnd", windowEnd)
						.addValue("homeTeam", normalizeForLookup(homeTeam))
						.addValue("awayTeam", normalizeForLookup(awayTeam)),
				EVENT_MATCH_CANDIDATE_ROW_MAPPER
		);

		return candidates.stream()
				.min(Comparator.comparingLong(candidate ->
						Math.abs(candidate.scheduledStart().toInstant().toEpochMilli() - commenceTime.toInstant().toEpochMilli())))
				.map(JdbcOddsRefreshRepository::toEventMatch);
	}

	@Override
	public Optional<EventMatch> findMlbEventByTeamCodes(LocalDate eventDate, String teamCode, String opponentCode) {
		OffsetDateTime dayStart = MlbTime.easternStartOfDay(eventDate);
		OffsetDateTime dayEnd = dayStart.plusDays(1);
		String normalizedTeamCode = MlbTeamMappings.canonicalCode(teamCode).toLowerCase(Locale.US);
		String normalizedOpponentCode = MlbTeamMappings.canonicalCode(opponentCode).toLowerCase(Locale.US);

		List<EventMatchCandidate> candidates = this.namedParameterJdbcTemplate.query(
				baseEventMatchSelect()
						+ """
						where lower(s.code) = 'mlb'
						  and e.scheduled_start >= :dayStart
						  and e.scheduled_start < :dayEnd
						  and (
						      (lower(home_t.code) = :teamCode and lower(away_t.code) = :opponentCode)
						      or
						      (lower(home_t.code) = :opponentCode and lower(away_t.code) = :teamCode)
						  )
						order by e.scheduled_start, e.id
						""",
				new MapSqlParameterSource()
						.addValue("dayStart", dayStart)
						.addValue("dayEnd", dayEnd)
						.addValue("teamCode", normalizedTeamCode)
						.addValue("opponentCode", normalizedOpponentCode),
				EVENT_MATCH_CANDIDATE_ROW_MAPPER
		);

		return candidates.stream().findFirst().map(JdbcOddsRefreshRepository::toEventMatch);
	}

	@Override
	public long ensureMlbSport() {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from sports
				where lower(code) = 'mlb'
				limit 1
				""",
				new MapSqlParameterSource(),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update sports
					set slug = :slug,
					    name = :name,
					    is_active = true,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("slug", "mlb")
							.addValue("name", "Major League Baseball")
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"insert into sports (code, slug, name, is_active) values (?, ?, ?, true)",
					new String[] {"id"}
			);
			statement.setString(1, "MLB");
			statement.setString(2, "mlb");
			statement.setString(3, "Major League Baseball");
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public long ensureMlbCompetition(long sportId) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from competitions
				where sport_id = :sportId
				  and lower(code) = 'mlb'
				limit 1
				""",
				new MapSqlParameterSource("sportId", sportId),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update competitions
					set slug = :slug,
					    name = :name,
					    competition_type = 'TEAM',
					    is_active = true,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("slug", "mlb")
							.addValue("name", "MLB Regular Season")
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into competitions (
					    sport_id, code, slug, name, competition_type, is_active
					) values (?, ?, ?, ?, 'TEAM', true)
					""",
					new String[] {"id"}
			);
			statement.setLong(1, sportId);
			statement.setString(2, "MLB");
			statement.setString(3, "mlb");
			statement.setString(4, "MLB Regular Season");
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public long upsertTeam(TeamUpsert team) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from teams
				where sport_id = :sportId
				  and (
				      external_ref = :externalRef
				      or lower(full_name) = :fullName
				  )
				order by case when external_ref = :externalRef then 0 else 1 end, id
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("sportId", team.sportId())
						.addValue("externalRef", team.externalRef())
						.addValue("fullName", normalizeForLookup(team.fullName())),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update teams
					set slug = :slug,
					    code = :code,
					    external_ref = :externalRef,
					    short_name = :shortName,
					    full_name = :fullName,
					    city = :city,
					    country_code = :countryCode,
					    is_active = true,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("slug", MlbTeamMappings.slugify(team.fullName()))
							.addValue("code", team.code())
							.addValue("externalRef", team.externalRef())
							.addValue("shortName", team.shortName())
							.addValue("fullName", team.fullName())
							.addValue("city", team.city())
							.addValue("countryCode", team.countryCode())
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into teams (
					    sport_id, slug, code, external_ref, short_name, full_name, city, country_code, is_active
					) values (?, ?, ?, ?, ?, ?, ?, ?, true)
					""",
					new String[] {"id"}
			);
			statement.setLong(1, team.sportId());
			statement.setString(2, MlbTeamMappings.slugify(team.fullName()));
			statement.setString(3, team.code());
			statement.setString(4, team.externalRef());
			statement.setString(5, team.shortName());
			statement.setString(6, team.fullName());
			statement.setString(7, team.city());
			statement.setString(8, team.countryCode());
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public void ensureCompetitionTeam(CompetitionTeamUpsert competitionTeam) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from competition_teams
				where competition_id = :competitionId
				  and team_id = :teamId
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("competitionId", competitionTeam.competitionId())
						.addValue("teamId", competitionTeam.teamId()),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			this.namedParameterJdbcTemplate.update(
					"""
					update competition_teams
					set external_ref = :externalRef,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", ids.get(0))
							.addValue("externalRef", competitionTeam.externalRef())
			);
			return;
		}

		this.namedParameterJdbcTemplate.update(
				"""
				insert into competition_teams (competition_id, team_id, external_ref)
				values (:competitionId, :teamId, :externalRef)
				""",
				new MapSqlParameterSource()
						.addValue("competitionId", competitionTeam.competitionId())
						.addValue("teamId", competitionTeam.teamId())
						.addValue("externalRef", competitionTeam.externalRef())
		);
	}

	@Override
	public EventMatch upsertMlbEvent(EventUpsert event) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from events
				where competition_id = :competitionId
				  and external_ref = :externalRef
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("competitionId", event.competitionId())
						.addValue("externalRef", event.externalRef()),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		long eventId;
		if (!ids.isEmpty()) {
			eventId = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update events
					set sport_id = :sportId,
					    slug = :slug,
					    external_ref = :externalRef,
					    event_type = :eventType,
					    status = :status,
					    season_label = :seasonLabel,
					    round_label = :roundLabel,
					    scheduled_start = :scheduledStart,
					    start_time_confirmed = :startTimeConfirmed,
					    venue_name = :venueName,
					    venue_city = :venueCity,
					    venue_country_code = :venueCountryCode,
					    updated_at = now()
					where id = :id
					""",
					eventParameters(event).addValue("id", eventId)
			);
		}
		else {
			KeyHolder keyHolder = new GeneratedKeyHolder();
			this.namedParameterJdbcTemplate.getJdbcTemplate().update(connection -> {
				PreparedStatement statement = connection.prepareStatement(
						"""
						insert into events (
						    sport_id,
						    competition_id,
						    slug,
						    external_ref,
						    event_type,
						    status,
						    season_label,
						    round_label,
						    scheduled_start,
						    start_time_confirmed,
						    venue_name,
						    venue_city,
						    venue_country_code
						) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
						""",
						new String[] {"id"}
				);
				statement.setLong(1, event.sportId());
				statement.setLong(2, event.competitionId());
				statement.setString(3, event.slug());
				statement.setString(4, event.externalRef());
				statement.setString(5, event.eventType());
				statement.setString(6, event.status());
				statement.setString(7, event.seasonLabel());
				statement.setString(8, event.roundLabel());
				statement.setObject(9, event.scheduledStart());
				statement.setBoolean(10, event.startTimeConfirmed());
				statement.setString(11, event.venueName());
				statement.setString(12, event.venueCity());
				statement.setString(13, event.venueCountryCode());
				return statement;
			}, keyHolder);
			eventId = keyHolder.getKey().longValue();
		}

		return new EventMatch(
				eventId,
				event.sportId(),
				MlbTime.toEasternDate(event.scheduledStart()),
				event.scheduledStart(),
				event.homeTeamId(),
				event.awayTeamId(),
				event.homeTeamCode(),
				event.awayTeamCode()
		);
	}

	@Override
	public long ensureTeamEventParticipant(TeamEventParticipantUpsert participant) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from event_participants
				where event_id = :eventId
				  and team_id = :teamId
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("eventId", participant.eventId())
						.addValue("teamId", participant.teamId()),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update event_participants
					set role_code = :roleCode,
					    sort_order = :sortOrder,
					    is_home = :isHome,
					    is_away = :isAway
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("roleCode", participant.roleCode())
							.addValue("sortOrder", participant.sortOrder())
							.addValue("isHome", participant.home())
							.addValue("isAway", participant.away())
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into event_participants (
					    event_id,
					    team_id,
					    role_code,
					    sort_order,
					    is_home,
					    is_away
					) values (?, ?, ?, ?, ?, ?)
					""",
					new String[] {"id"}
			);
			statement.setLong(1, participant.eventId());
			statement.setLong(2, participant.teamId());
			statement.setString(3, participant.roleCode());
			statement.setInt(4, participant.sortOrder());
			statement.setBoolean(5, participant.home());
			statement.setBoolean(6, participant.away());
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public long upsertSportsbook(SportsbookUpsert sportsbook) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from sportsbooks
				where lower(code) = :code
				""",
				new MapSqlParameterSource("code", sportsbook.code().toLowerCase(Locale.US)),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update sportsbooks
					set slug = :slug,
					    display_name = :displayName,
					    region_code = :regionCode,
					    is_active = true,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("slug", sportsbook.slug())
							.addValue("displayName", sportsbook.displayName())
							.addValue("regionCode", sportsbook.regionCode())
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into sportsbooks (code, slug, display_name, region_code, is_active)
					values (?, ?, ?, ?, true)
					""",
					new String[] {"id"}
			);
			statement.setString(1, sportsbook.code());
			statement.setString(2, sportsbook.slug());
			statement.setString(3, sportsbook.displayName());
			statement.setString(4, sportsbook.regionCode());
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public long upsertPitcherStrikeoutsMarket(long sportId) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from markets
				where sport_id = :sportId
				  and competition_id is null
				  and lower(code) = 'pitcher_strikeouts'
				""",
				new MapSqlParameterSource("sportId", sportId),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update markets
					set slug = :slug,
					    display_name = :displayName,
					    market_scope = 'PLAYER',
					    stat_type = 'STRIKEOUTS',
					    period_type = 'FULL_EVENT',
					    allows_over_under = true,
					    allows_binary_outcome = false,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("slug", "pitcher-strikeouts")
							.addValue("displayName", "Pitcher Strikeouts")
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into markets (
					    sport_id,
					    competition_id,
					    code,
					    slug,
					    display_name,
					    market_scope,
					    stat_type,
					    period_type,
					    allows_over_under,
					    allows_binary_outcome
					) values (?, null, ?, ?, ?, 'PLAYER', 'STRIKEOUTS', 'FULL_EVENT', true, false)
					""",
					new String[] {"id"}
			);
			statement.setLong(1, sportId);
			statement.setString(2, "PITCHER_STRIKEOUTS");
			statement.setString(3, "pitcher-strikeouts");
			statement.setString(4, "Pitcher Strikeouts");
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public long upsertPlayer(PlayerUpsert player) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from players
				where sport_id = :sportId
				  and (
				      external_ref = :externalRef
				      or lower(display_name) = :displayName
				  )
				order by case when external_ref = :externalRef then 0 else 1 end, id
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("sportId", player.sportId())
						.addValue("externalRef", player.externalRef())
						.addValue("displayName", player.displayName().toLowerCase(Locale.US)),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		String[] nameParts = splitName(player.displayName());
		if (!ids.isEmpty()) {
			long id = ids.get(0);
			this.namedParameterJdbcTemplate.update(
					"""
					update players
					set external_ref = :externalRef,
					    display_name = :displayName,
					    first_name = :firstName,
					    last_name = :lastName,
					    is_active = true,
					    updated_at = now()
					where id = :id
					""",
					new MapSqlParameterSource()
							.addValue("id", id)
							.addValue("externalRef", player.externalRef())
							.addValue("displayName", player.displayName())
							.addValue("firstName", nameParts[0])
							.addValue("lastName", nameParts[1])
			);
			return id;
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into players (
					    sport_id,
					    slug,
					    external_ref,
					    first_name,
					    last_name,
					    display_name,
					    is_active
					) values (?, ?, ?, ?, ?, ?, true)
					""",
					new String[] {"id"}
			);
			statement.setLong(1, player.sportId());
			statement.setString(2, slugify(player.displayName(), player.externalRef()));
			statement.setString(3, player.externalRef());
			statement.setString(4, nameParts[0]);
			statement.setString(5, nameParts[1]);
			statement.setString(6, player.displayName());
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public long ensurePlayerEventParticipant(PlayerEventParticipantUpsert participant) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from event_participants
				where event_id = :eventId
				  and player_id = :playerId
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("eventId", participant.eventId())
						.addValue("playerId", participant.playerId()),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			return ids.get(0);
		}

		KeyHolder keyHolder = new GeneratedKeyHolder();
		this.jdbcTemplate.update(connection -> {
			PreparedStatement statement = connection.prepareStatement(
					"""
					insert into event_participants (
					    event_id,
					    player_id,
					    role_code,
					    sort_order,
					    is_home,
					    is_away
					) values (?, ?, ?, ?, ?, ?)
					""",
					new String[] {"id"}
			);
			statement.setLong(1, participant.eventId());
			statement.setLong(2, participant.playerId());
			statement.setString(3, participant.roleCode());
			statement.setInt(4, participant.sortOrder());
			statement.setBoolean(5, participant.home());
			statement.setBoolean(6, participant.away());
			return statement;
		}, keyHolder);
		return keyHolder.getKey().longValue();
	}

	@Override
	public void upsertOffer(OfferUpsert offer) {
		List<Long> ids = this.namedParameterJdbcTemplate.query(
				"""
				select id
				from offers
				where sportsbook_id = :sportsbookId
				  and source_offer_id = :sourceOfferId
				limit 1
				""",
				new MapSqlParameterSource()
						.addValue("sportsbookId", offer.sportsbookId())
						.addValue("sourceOfferId", offer.sourceOfferId()),
				(resultSet, rowNum) -> resultSet.getLong("id")
		);

		if (!ids.isEmpty()) {
			this.namedParameterJdbcTemplate.update(
					"""
					update offers
					set event_id = :eventId,
					    market_id = :marketId,
					    event_participant_id = :eventParticipantId,
					    line_value = :lineValue,
					    price_american = :americanPrice,
					    price_decimal = :decimalPrice,
					    selection_label = :selectionLabel,
					    side_code = :sideCode,
					    outcome_type = :outcomeType,
					    available_at = :availableAt,
					    is_live = :isLive,
					    updated_at = now()
					where id = :id
					""",
					offerParameters(offer).addValue("id", ids.get(0))
			);
			return;
		}

		this.namedParameterJdbcTemplate.update(
				"""
				insert into offers (
				    sportsbook_id,
				    event_id,
				    market_id,
				    event_participant_id,
				    line_value,
				    price_american,
				    price_decimal,
				    selection_label,
				    side_code,
				    outcome_type,
				    available_at,
				    is_live,
				    source_offer_id
				) values (
				    :sportsbookId,
				    :eventId,
				    :marketId,
				    :eventParticipantId,
				    :lineValue,
				    :americanPrice,
				    :decimalPrice,
				    :selectionLabel,
				    :sideCode,
				    :outcomeType,
				    :availableAt,
				    :isLive,
				    :sourceOfferId
				)
				""",
				offerParameters(offer)
		);
	}

	@Override
	public int deleteMissingOffers(long sportsbookId, long eventId, long marketId, Set<String> retainedSourceOfferIds) {
		MapSqlParameterSource parameters = new MapSqlParameterSource()
				.addValue("sportsbookId", sportsbookId)
				.addValue("eventId", eventId)
				.addValue("marketId", marketId);

		if (retainedSourceOfferIds.isEmpty()) {
			return this.namedParameterJdbcTemplate.update(
					"""
					delete from offers
					where sportsbook_id = :sportsbookId
					  and event_id = :eventId
					  and market_id = :marketId
					  and source_offer_id is not null
					  and source_offer_id not like 'tracked|%'
					""",
					parameters
			);
		}

		return this.namedParameterJdbcTemplate.update(
				"""
				delete from offers
				where sportsbook_id = :sportsbookId
				  and event_id = :eventId
				  and market_id = :marketId
				  and source_offer_id is not null
				  and source_offer_id not like 'tracked|%'
				  and source_offer_id not in (:sourceOfferIds)
				""",
				parameters.addValue("sourceOfferIds", retainedSourceOfferIds)
		);
	}

	private static String baseEventMatchSelect() {
		return """
				select
				    e.id as event_id,
				    e.sport_id,
				    e.scheduled_start,
				    home_t.id as home_team_id,
				    away_t.id as away_team_id,
				    home_t.code as home_team_code,
				    away_t.code as away_team_code
				from events e
				join sports s on s.id = e.sport_id
				join event_participants home_ep on home_ep.event_id = e.id and home_ep.is_home = true
				join teams home_t on home_t.id = home_ep.team_id
				join event_participants away_ep on away_ep.event_id = e.id and away_ep.is_away = true
				join teams away_t on away_t.id = away_ep.team_id
				""";
	}

	private static EventMatch toEventMatch(EventMatchCandidate candidate) {
		return new EventMatch(
				candidate.eventId(),
				candidate.sportId(),
				MlbTime.toEasternDate(candidate.scheduledStart()),
				candidate.scheduledStart(),
				candidate.homeTeamId(),
				candidate.awayTeamId(),
				candidate.homeTeamCode(),
				candidate.awayTeamCode()
		);
	}

	private static MapSqlParameterSource eventParameters(EventUpsert event) {
		return new MapSqlParameterSource()
				.addValue("sportId", event.sportId())
				.addValue("competitionId", event.competitionId())
				.addValue("slug", event.slug())
				.addValue("externalRef", event.externalRef())
				.addValue("eventType", event.eventType())
				.addValue("status", event.status())
				.addValue("seasonLabel", event.seasonLabel())
				.addValue("roundLabel", event.roundLabel())
				.addValue("scheduledStart", event.scheduledStart())
				.addValue("startTimeConfirmed", event.startTimeConfirmed())
				.addValue("venueName", event.venueName())
				.addValue("venueCity", event.venueCity())
				.addValue("venueCountryCode", event.venueCountryCode());
	}

	private static MapSqlParameterSource offerParameters(OfferUpsert offer) {
		return new MapSqlParameterSource()
				.addValue("sportsbookId", offer.sportsbookId())
				.addValue("eventId", offer.eventId())
				.addValue("marketId", offer.marketId())
				.addValue("eventParticipantId", offer.eventParticipantId())
				.addValue("lineValue", offer.lineValue())
				.addValue("americanPrice", offer.americanPrice())
				.addValue("decimalPrice", offer.decimalPrice())
				.addValue("selectionLabel", offer.selectionLabel())
				.addValue("sideCode", offer.sideCode())
				.addValue("outcomeType", offer.outcomeType())
				.addValue("availableAt", offer.availableAt())
				.addValue("isLive", offer.live())
				.addValue("sourceOfferId", offer.sourceOfferId());
	}

	private static String[] splitName(String displayName) {
		String trimmed = displayName == null ? "" : displayName.trim();
		if (trimmed.isEmpty()) {
			return new String[] {null, null};
		}
		int lastSpace = trimmed.lastIndexOf(' ');
		if (lastSpace < 0) {
			return new String[] {trimmed, null};
		}
		return new String[] {
				trimmed.substring(0, lastSpace).trim(),
				trimmed.substring(lastSpace + 1).trim()
		};
	}

	private static String slugify(String displayName, String externalRef) {
		String normalized = Normalizer.normalize(displayName, Normalizer.Form.NFD)
				.replaceAll("\\p{M}", "")
				.toLowerCase(Locale.US)
				.replaceAll("[^a-z0-9]+", "-")
				.replaceAll("(^-|-$)", "");
		String suffix = externalRef == null ? "" : externalRef.replaceAll("[^a-zA-Z0-9]+", "-").toLowerCase(Locale.US);
		if (suffix.isEmpty()) {
			return normalized;
		}
		return normalized + "-" + suffix;
	}

	private static String normalizeForLookup(String value) {
		return value == null ? "" : value.trim().toLowerCase(Locale.US);
	}

	private record EventMatchCandidate(
			long eventId,
			long sportId,
			OffsetDateTime scheduledStart,
			long homeTeamId,
			long awayTeamId,
			String homeTeamCode,
			String awayTeamCode
	) {
	}
}
