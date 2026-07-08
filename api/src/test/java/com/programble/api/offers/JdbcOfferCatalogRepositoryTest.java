package com.programble.api.offers;

import java.math.BigDecimal;
import java.sql.ResultSet;
import java.time.OffsetDateTime;

import com.programble.api.offers.OfferCatalogService.ProgrambleOffer;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class JdbcOfferCatalogRepositoryTest {

	@Test
	void mapsOffersWhenJdbcNumericObjectsAreReturnedAsNumbers() throws Exception {
		ResultSet resultSet = mock(ResultSet.class);
		OffsetDateTime scheduledStart = OffsetDateTime.parse("2026-07-06T23:05:00Z");
		OffsetDateTime availableAt = OffsetDateTime.parse("2026-07-06T20:15:00Z");
patched the most likely PostgreSQL-only failure point in the offer mapper by removing the remaining unsafe JDBC getObject() numeric casts in [JdbcOfferCatalogRepository.java](/C:/ProGramble/proGramble/api/src/main/java/com/programble/api/offers/JdbcOfferCatalogRepository.java) and [JdbcEventCatalogRepository.java](/C:/ProGramble/proGramble/api/src/main/java/com/programble/api/events/JdbcEventCatalogRepository.java), and added a regression test in [JdbcOfferCatalogRepositoryTest.java](/C:/ProGramble/proGramble/api/src/test/java/com/programble/api/offers/JdbcOfferCatalogRepositoryTest.java)
		when(resultSet.getLong("offer_id")).thenReturn(7001L);
		when(resultSet.getLong("event_id")).thenReturn(405L);
		when(resultSet.getLong("sport_id")).thenReturn(1L);
		when(resultSet.getLong("competition_id")).thenReturn(11L);
		when(resultSet.getLong("sportsbook_id")).thenReturn(601L);
		when(resultSet.getLong("market_id")).thenReturn(502L);

		when(resultSet.getObject("event_participant_id")).thenReturn(9101L);
		when(resultSet.getObject("team_id")).thenReturn(null);
		when(resultSet.getObject("player_id")).thenReturn(301L);
		when(resultSet.getObject("sort_order")).thenReturn(3);
		when(resultSet.getObject("is_home")).thenReturn(Boolean.FALSE);
		when(resultSet.getObject("is_away")).thenReturn(Boolean.FALSE);
		when(resultSet.getObject("price_american")).thenReturn(-115);
		when(resultSet.getObject(eq("scheduled_start"), eq(OffsetDateTime.class))).thenReturn(scheduledStart);
		when(resultSet.getObject(eq("available_at"), eq(OffsetDateTime.class))).thenReturn(availableAt);

		when(resultSet.getString("event_slug")).thenReturn("yankees-red-sox-2026-07-06");
		when(resultSet.getString("event_external_ref")).thenReturn("mlb-20260706-nyy-bos");
		when(resultSet.getString("event_type")).thenReturn("GAME");
		when(resultSet.getString("event_status")).thenReturn("SCHEDULED");
		when(resultSet.getString("sport_code")).thenReturn("MLB");
		when(resultSet.getString("sport_slug")).thenReturn("mlb");
		when(resultSet.getString("sport_name")).thenReturn("Major League Baseball");
		when(resultSet.getString("competition_code")).thenReturn("MLB");
		when(resultSet.getString("competition_slug")).thenReturn("mlb");
		when(resultSet.getString("competition_name")).thenReturn("MLB Regular Season");
		when(resultSet.getString("competition_type")).thenReturn("TEAM");
		when(resultSet.getString("venue_name")).thenReturn("Fenway Park");
		when(resultSet.getString("venue_city")).thenReturn("Boston");
		when(resultSet.getString("venue_country_code")).thenReturn("US");
		when(resultSet.getString("sportsbook_code")).thenReturn("DK");
		when(resultSet.getString("sportsbook_slug")).thenReturn("draftkings");
		when(resultSet.getString("sportsbook_display_name")).thenReturn("DraftKings");
		when(resultSet.getString("sportsbook_region_code")).thenReturn("US");
		when(resultSet.getString("market_code")).thenReturn("PITCHER_STRIKEOUTS");
		when(resultSet.getString("market_slug")).thenReturn("pitcher-strikeouts");
		when(resultSet.getString("market_display_name")).thenReturn("Pitcher Strikeouts");
		when(resultSet.getString("market_scope")).thenReturn("PLAYER");
		when(resultSet.getString("stat_type")).thenReturn("PITCHER_STRIKEOUTS");
		when(resultSet.getString("period_type")).thenReturn("FULL_EVENT");
		when(resultSet.getString("participant_type")).thenReturn("PLAYER");
		when(resultSet.getString("role_code")).thenReturn("PROBABLE_PITCHER_AWAY");
		when(resultSet.getString("participant_display_name")).thenReturn("Gerrit Cole");
		when(resultSet.getString("participant_short_name")).thenReturn("Gerrit Cole");
		when(resultSet.getString("selection_label")).thenReturn("Gerrit Cole Over 6.5 Strikeouts");
		when(resultSet.getString("side_code")).thenReturn("OVER");
		when(resultSet.getString("outcome_type")).thenReturn("PROP");

		when(resultSet.getBigDecimal("line_value")).thenReturn(new BigDecimal("6.5"));
		when(resultSet.getBigDecimal("price_decimal")).thenReturn(new BigDecimal("1.8696"));
		when(resultSet.getBoolean("start_time_confirmed")).thenReturn(true);
		when(resultSet.getBoolean("allows_over_under")).thenReturn(true);
		when(resultSet.getBoolean("allows_binary_outcome")).thenReturn(false);
		when(resultSet.getBoolean("is_live")).thenReturn(false);

		ProgrambleOffer offer = JdbcOfferCatalogRepository.mapOffer(resultSet);

		assertThat(offer.id()).isEqualTo(7001L);
		assertThat(offer.event().id()).isEqualTo(405L);
		assertThat(offer.market().code()).isEqualTo("PITCHER_STRIKEOUTS");
		assertThat(offer.participant()).isNotNull();
		assertThat(offer.participant().eventParticipantId()).isEqualTo(9101L);
		assertThat(offer.participant().playerId()).isEqualTo(301L);
		assertThat(offer.participant().teamId()).isNull();
		assertThat(offer.participant().sortOrder()).isEqualTo(3);
		assertThat(offer.selectionLabel()).isEqualTo("Gerrit Cole Over 6.5 Strikeouts");
		assertThat(offer.availableAt()).isEqualTo(availableAt);
	}

	@Test
	void mapsOffersWhenJdbcNumericObjectsUseMixedNumberImplementations() throws Exception {
		ResultSet resultSet = mock(ResultSet.class);
		OffsetDateTime scheduledStart = OffsetDateTime.parse("2026-07-08T22:40:00Z");
		OffsetDateTime availableAt = OffsetDateTime.parse("2026-07-08T14:35:00Z");

		when(resultSet.getLong("offer_id")).thenReturn(9001L);
		when(resultSet.getLong("event_id")).thenReturn(640L);
		when(resultSet.getLong("sport_id")).thenReturn(1L);
		when(resultSet.getLong("competition_id")).thenReturn(11L);
		when(resultSet.getLong("sportsbook_id")).thenReturn(601L);
		when(resultSet.getLong("market_id")).thenReturn(502L);

		when(resultSet.getObject("event_participant_id")).thenReturn(Long.valueOf(9201L));
		when(resultSet.getObject("team_id")).thenReturn(null);
		when(resultSet.getObject("player_id")).thenReturn(Long.valueOf(555L));
		when(resultSet.getObject("sort_order")).thenReturn(Long.valueOf(10L));
		when(resultSet.getObject("is_home")).thenReturn(Boolean.FALSE);
		when(resultSet.getObject("is_away")).thenReturn(Boolean.TRUE);
		when(resultSet.getObject("price_american")).thenReturn(Long.valueOf(-120L));
		when(resultSet.getObject(eq("scheduled_start"), eq(OffsetDateTime.class))).thenReturn(scheduledStart);
		when(resultSet.getObject(eq("available_at"), eq(OffsetDateTime.class))).thenReturn(availableAt);

		when(resultSet.getString("event_slug")).thenReturn("atlanta-braves-pittsburgh-pirates-2026-07-08-823360");
		when(resultSet.getString("event_external_ref")).thenReturn("mlb_stats_api_game:823360");
		when(resultSet.getString("event_type")).thenReturn("GAME");
		when(resultSet.getString("event_status")).thenReturn("SCHEDULED");
		when(resultSet.getString("sport_code")).thenReturn("MLB");
		when(resultSet.getString("sport_slug")).thenReturn("mlb");
		when(resultSet.getString("sport_name")).thenReturn("Major League Baseball");
		when(resultSet.getString("competition_code")).thenReturn("MLB");
		when(resultSet.getString("competition_slug")).thenReturn("mlb");
		when(resultSet.getString("competition_name")).thenReturn("MLB Regular Season");
		when(resultSet.getString("competition_type")).thenReturn("TEAM");
		when(resultSet.getString("venue_name")).thenReturn("PNC Park");
		when(resultSet.getString("venue_city")).thenReturn("Pittsburgh");
		when(resultSet.getString("venue_country_code")).thenReturn("US");
		when(resultSet.getString("sportsbook_code")).thenReturn("DK");
		when(resultSet.getString("sportsbook_slug")).thenReturn("draftkings");
		when(resultSet.getString("sportsbook_display_name")).thenReturn("DraftKings");
		when(resultSet.getString("sportsbook_region_code")).thenReturn("US");
		when(resultSet.getString("market_code")).thenReturn("PITCHER_STRIKEOUTS");
		when(resultSet.getString("market_slug")).thenReturn("pitcher-strikeouts");
		when(resultSet.getString("market_display_name")).thenReturn("Pitcher Strikeouts");
		when(resultSet.getString("market_scope")).thenReturn("PLAYER");
		when(resultSet.getString("stat_type")).thenReturn("STRIKEOUTS");
		when(resultSet.getString("period_type")).thenReturn("FULL_EVENT");
		when(resultSet.getString("participant_type")).thenReturn("PLAYER");
		when(resultSet.getString("role_code")).thenReturn("STARTING_PITCHER_AWAY");
		when(resultSet.getString("participant_display_name")).thenReturn("Spencer Strider");
		when(resultSet.getString("participant_short_name")).thenReturn("Spencer Strider");
		when(resultSet.getString("selection_label")).thenReturn("Spencer Strider Over 6.5 Strikeouts");
		when(resultSet.getString("side_code")).thenReturn("OVER");
		when(resultSet.getString("outcome_type")).thenReturn("PROP");

		when(resultSet.getBigDecimal("line_value")).thenReturn(new BigDecimal("6.5"));
		when(resultSet.getBigDecimal("price_decimal")).thenReturn(new BigDecimal("1.8333"));
		when(resultSet.getBoolean("start_time_confirmed")).thenReturn(true);
		when(resultSet.getBoolean("allows_over_under")).thenReturn(true);
		when(resultSet.getBoolean("allows_binary_outcome")).thenReturn(false);
		when(resultSet.getBoolean("is_live")).thenReturn(false);

		ProgrambleOffer offer = JdbcOfferCatalogRepository.mapOffer(resultSet);

		assertThat(offer.participant()).isNotNull();
		assertThat(offer.participant().sortOrder()).isEqualTo(10);
		assertThat(offer.priceAmerican()).isEqualTo(-120);
		assertThat(offer.participant().playerId()).isEqualTo(555L);
	}
}
