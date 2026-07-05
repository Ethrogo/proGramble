package com.programble.api.offers;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;

import com.programble.api.offers.OfferCatalogService.CompetitionDescriptor;
import com.programble.api.offers.OfferCatalogService.EventDescriptor;
import com.programble.api.offers.OfferCatalogService.MarketDescriptor;
import com.programble.api.offers.OfferCatalogService.OfferParticipantDescriptor;
import com.programble.api.offers.OfferCatalogService.PlayerDescriptor;
import com.programble.api.offers.OfferCatalogService.ProgrambleOffer;
import com.programble.api.offers.OfferCatalogService.SportDescriptor;
import com.programble.api.offers.OfferCatalogService.SportsbookDescriptor;
import com.programble.api.offers.OfferCatalogService.VenueDescriptor;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class JdbcOfferCatalogRepository implements OfferCatalogRepository {

	private final NamedParameterJdbcTemplate jdbcTemplate;

	public JdbcOfferCatalogRepository(NamedParameterJdbcTemplate jdbcTemplate) {
		this.jdbcTemplate = jdbcTemplate;
	}

	@Override
	public Optional<PlayerDescriptor> findPlayer(long playerId) {
		var players = this.jdbcTemplate.query(
				"""
				select
				    p.id,
				    p.slug,
				    p.display_name,
				    s.id as sport_id,
				    s.code as sport_code,
				    s.slug as sport_slug,
				    s.name as sport_name
				from players p
				join sports s on s.id = p.sport_id
				where p.id = :playerId
				""",
				new MapSqlParameterSource("playerId", playerId),
				(resultSet, rowNum) -> new PlayerDescriptor(
						resultSet.getLong("id"),
						resultSet.getString("slug"),
						resultSet.getString("display_name"),
						new SportDescriptor(
								resultSet.getLong("sport_id"),
								resultSet.getString("sport_code"),
								resultSet.getString("sport_slug"),
								resultSet.getString("sport_name")
						)
				)
		);

		return players.stream().findFirst();
	}

	@Override
	public List<ProgrambleOffer> findOffersForEvent(long eventId, Long playerId, String sportsbookKey, String marketTypeKey) {
		return this.jdbcTemplate.query(
				"""
				select
				    o.id as offer_id,
				    o.line_value,
				    o.price_american,
				    o.price_decimal,
				    o.selection_label,
				    o.side_code,
				    o.outcome_type,
				    o.available_at,
				    o.is_live,
				    sb.id as sportsbook_id,
				    sb.code as sportsbook_code,
				    sb.slug as sportsbook_slug,
				    sb.display_name as sportsbook_display_name,
				    sb.region_code as sportsbook_region_code,
				    m.id as market_id,
				    m.code as market_code,
				    m.slug as market_slug,
				    m.display_name as market_display_name,
				    m.market_scope,
				    m.stat_type,
				    m.period_type,
				    m.allows_over_under,
				    m.allows_binary_outcome,
				    e.id as event_id,
				    e.slug as event_slug,
				    e.external_ref as event_external_ref,
				    e.event_type,
				    e.status as event_status,
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
				    c.competition_type,
				    ep.id as event_participant_id,
				    ep.team_id,
				    ep.player_id,
				    ep.role_code,
				    ep.sort_order,
				    ep.is_home,
				    ep.is_away,
				    case
				        when ep.id is null then null
				        when ep.team_id is not null then 'TEAM'
				        else 'PLAYER'
				    end as participant_type,
				    coalesce(t.full_name, p.display_name) as participant_display_name,
				    coalesce(t.short_name, p.display_name) as participant_short_name
				from offers o
				join sportsbooks sb on sb.id = o.sportsbook_id
				join markets m on m.id = o.market_id
				join events e on e.id = o.event_id
				join sports s on s.id = e.sport_id
				join competitions c on c.id = e.competition_id
				left join event_participants ep on ep.id = o.event_participant_id
				left join teams t on t.id = ep.team_id
				left join players p on p.id = ep.player_id
				where o.event_id = :eventId
				  and (:playerId is null or ep.player_id = :playerId)
				  and (
				        :sportsbookKey is null
				        or lower(sb.code) = :sportsbookKey
				        or lower(sb.slug) = :sportsbookKey
				  )
				  and (
				        :marketTypeKey is null
				        or lower(m.code) = :marketTypeKey
				        or lower(m.slug) = :marketTypeKey
				        or lower(coalesce(m.stat_type, '')) = :marketTypeKey
				  )
				order by
				    m.display_name,
				    coalesce(ep.sort_order, 9999),
				    sb.display_name,
				    o.selection_label,
				    o.available_at desc,
				    o.id
				""",
				new MapSqlParameterSource()
						.addValue("eventId", eventId)
						.addValue("playerId", playerId)
						.addValue("sportsbookKey", sportsbookKey)
						.addValue("marketTypeKey", marketTypeKey),
				(resultSet, rowNum) -> mapOffer(resultSet)
		);
	}

	@Override
	public List<ProgrambleOffer> findOffersForPlayer(long playerId, String sportsbookKey, String marketTypeKey) {
		return this.jdbcTemplate.query(
				"""
				select
				    o.id as offer_id,
				    o.line_value,
				    o.price_american,
				    o.price_decimal,
				    o.selection_label,
				    o.side_code,
				    o.outcome_type,
				    o.available_at,
				    o.is_live,
				    sb.id as sportsbook_id,
				    sb.code as sportsbook_code,
				    sb.slug as sportsbook_slug,
				    sb.display_name as sportsbook_display_name,
				    sb.region_code as sportsbook_region_code,
				    m.id as market_id,
				    m.code as market_code,
				    m.slug as market_slug,
				    m.display_name as market_display_name,
				    m.market_scope,
				    m.stat_type,
				    m.period_type,
				    m.allows_over_under,
				    m.allows_binary_outcome,
				    e.id as event_id,
				    e.slug as event_slug,
				    e.external_ref as event_external_ref,
				    e.event_type,
				    e.status as event_status,
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
				    c.competition_type,
				    ep.id as event_participant_id,
				    ep.team_id,
				    ep.player_id,
				    ep.role_code,
				    ep.sort_order,
				    ep.is_home,
				    ep.is_away,
				    case
				        when ep.team_id is not null then 'TEAM'
				        else 'PLAYER'
				    end as participant_type,
				    coalesce(t.full_name, p.display_name) as participant_display_name,
				    coalesce(t.short_name, p.display_name) as participant_short_name
				from offers o
				join sportsbooks sb on sb.id = o.sportsbook_id
				join markets m on m.id = o.market_id
				join events e on e.id = o.event_id
				join sports s on s.id = e.sport_id
				join competitions c on c.id = e.competition_id
				join event_participants ep on ep.id = o.event_participant_id
				left join teams t on t.id = ep.team_id
				left join players p on p.id = ep.player_id
				where ep.player_id = :playerId
				  and (
				        :sportsbookKey is null
				        or lower(sb.code) = :sportsbookKey
				        or lower(sb.slug) = :sportsbookKey
				  )
				  and (
				        :marketTypeKey is null
				        or lower(m.code) = :marketTypeKey
				        or lower(m.slug) = :marketTypeKey
				        or lower(coalesce(m.stat_type, '')) = :marketTypeKey
				  )
				order by
				    e.scheduled_start,
				    m.display_name,
				    sb.display_name,
				    o.selection_label,
				    o.available_at desc,
				    o.id
				""",
				new MapSqlParameterSource()
						.addValue("playerId", playerId)
						.addValue("sportsbookKey", sportsbookKey)
						.addValue("marketTypeKey", marketTypeKey),
				(resultSet, rowNum) -> mapOffer(resultSet)
		);
	}

	private static ProgrambleOffer mapOffer(ResultSet resultSet) throws SQLException {
		Long eventParticipantId = (Long) resultSet.getObject("event_participant_id");
		OfferParticipantDescriptor participant = null;

		if (eventParticipantId != null) {
			participant = new OfferParticipantDescriptor(
					eventParticipantId,
					(Long) resultSet.getObject("team_id"),
					(Long) resultSet.getObject("player_id"),
					resultSet.getString("participant_type"),
					resultSet.getString("role_code"),
					resultSet.getString("participant_display_name"),
					resultSet.getString("participant_short_name"),
					(Integer) resultSet.getObject("sort_order"),
					(Boolean) resultSet.getObject("is_home"),
					(Boolean) resultSet.getObject("is_away")
			);
		}

		return new ProgrambleOffer(
				resultSet.getLong("offer_id"),
				new EventDescriptor(
						resultSet.getLong("event_id"),
						resultSet.getString("event_slug"),
						resultSet.getString("event_external_ref"),
						resultSet.getString("event_type"),
						resultSet.getString("event_status"),
						resultSet.getObject("scheduled_start", java.time.OffsetDateTime.class),
						resultSet.getBoolean("start_time_confirmed"),
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
						new VenueDescriptor(
								resultSet.getString("venue_name"),
								resultSet.getString("venue_city"),
								resultSet.getString("venue_country_code")
						)
				),
				new SportsbookDescriptor(
						resultSet.getLong("sportsbook_id"),
						resultSet.getString("sportsbook_code"),
						resultSet.getString("sportsbook_slug"),
						resultSet.getString("sportsbook_display_name"),
						resultSet.getString("sportsbook_region_code")
				),
				new MarketDescriptor(
						resultSet.getLong("market_id"),
						resultSet.getString("market_code"),
						resultSet.getString("market_slug"),
						resultSet.getString("market_display_name"),
						resultSet.getString("market_scope"),
						resultSet.getString("stat_type"),
						resultSet.getString("period_type"),
						resultSet.getBoolean("allows_over_under"),
						resultSet.getBoolean("allows_binary_outcome")
				),
				participant,
				resultSet.getBigDecimal("line_value"),
				(Integer) resultSet.getObject("price_american"),
				resultSet.getBigDecimal("price_decimal"),
				resultSet.getString("selection_label"),
				resultSet.getString("side_code"),
				resultSet.getString("outcome_type"),
				resultSet.getObject("available_at", java.time.OffsetDateTime.class),
				resultSet.getBoolean("is_live")
		);
	}
}
