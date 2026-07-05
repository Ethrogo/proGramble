package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.programble.api.config.OddsApiProperties;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@Component
public class HttpOddsApiClient implements OddsApiClient {

	private static final ParameterizedTypeReference<List<OddsApiEventSummary>> EVENT_LIST_TYPE = new ParameterizedTypeReference<>() {
	};

	private final OddsApiProperties properties;
	private final RestTemplate restTemplate;

	public HttpOddsApiClient(OddsApiProperties properties, RestTemplateBuilder restTemplateBuilder) {
		this.properties = properties;
		this.restTemplate = restTemplateBuilder.build();
	}

	@Override
	public List<MlbPitcherStrikeoutEvent> fetchMlbPitcherStrikeoutEvents() {
		String apiKey = this.properties.apiKey();
		if (!StringUtils.hasText(apiKey)) {
			throw new IllegalStateException("PROGRAMBLE_ODDS_API_KEY or ODDS_API_KEY must be configured for refresh-odds.");
		}

		ResponseEntity<List<OddsApiEventSummary>> eventResponse = this.restTemplate.exchange(
				UriComponentsBuilder.fromHttpUrl(this.properties.baseUrl())
						.pathSegment(this.properties.mlbSportKey(), "odds")
						.queryParam("apiKey", apiKey)
						.queryParam("regions", "us")
						.queryParam("markets", this.properties.eventDiscoveryMarketKey())
						.queryParam("oddsFormat", "american")
						.queryParam("dateFormat", "iso")
						.toUriString(),
				org.springframework.http.HttpMethod.GET,
				null,
				EVENT_LIST_TYPE
		);

		List<OddsApiEventSummary> events = eventResponse.getBody() == null ? List.of() : eventResponse.getBody();
		List<MlbPitcherStrikeoutEvent> propEvents = new ArrayList<>();

		for (OddsApiEventSummary event : events) {
			if (!StringUtils.hasText(event.id())) {
				continue;
			}

			OddsApiEventDetail detail = fetchEventDetail(apiKey, event.id(), this.properties.defaultBookmakers());
			if (detail == null) {
				continue;
			}

			List<PitcherStrikeoutOffer> offers = extractOffers(detail);
			if (offers.isEmpty() && !this.properties.defaultBookmakers().isEmpty()) {
				OddsApiEventDetail fallbackDetail = fetchEventDetail(apiKey, event.id(), List.of());
				if (fallbackDetail != null) {
					detail = fallbackDetail;
					offers = extractOffers(detail);
				}
			}

			propEvents.add(new MlbPitcherStrikeoutEvent(
					detail.id(),
					detail.commenceTime(),
					detail.homeTeam(),
					detail.awayTeam(),
					List.copyOf(offers)
			));
		}

		return List.copyOf(propEvents);
	}

	private OddsApiEventDetail fetchEventDetail(String apiKey, String eventId, List<String> bookmakers) {
		UriComponentsBuilder uriBuilder = UriComponentsBuilder.fromHttpUrl(this.properties.baseUrl())
				.pathSegment(this.properties.mlbSportKey(), "events", eventId, "odds")
				.queryParam("apiKey", apiKey)
				.queryParam("regions", "us")
				.queryParam("markets", this.properties.mlbPitcherStrikeoutsMarketKey())
				.queryParam("oddsFormat", "american")
				.queryParam("dateFormat", "iso");
		if (!bookmakers.isEmpty()) {
			uriBuilder.queryParam("bookmakers", String.join(",", bookmakers));
		}

		ResponseEntity<OddsApiEventDetail> detailResponse = this.restTemplate.getForEntity(
				uriBuilder.toUriString(),
				OddsApiEventDetail.class
		);
		return detailResponse.getBody();
	}

	private List<PitcherStrikeoutOffer> extractOffers(OddsApiEventDetail detail) {
		List<PitcherStrikeoutOffer> offers = new ArrayList<>();
		if (detail.bookmakers() == null) {
			return offers;
		}

		for (OddsApiBookmaker bookmaker : detail.bookmakers()) {
			if (bookmaker == null || !StringUtils.hasText(bookmaker.key()) || bookmaker.markets() == null) {
				continue;
			}

			String sportsbookKey = bookmaker.key();
			String sportsbookDisplayName = StringUtils.hasText(bookmaker.title()) ? bookmaker.title() : sportsbookKey;
			OffsetDateTime availableAt = bookmaker.lastUpdate();

			for (OddsApiMarket market : bookmaker.markets()) {
				if (market == null
						|| !this.properties.mlbPitcherStrikeoutsMarketKey().equals(market.key())
						|| market.outcomes() == null) {
					continue;
				}

				for (OddsApiOutcome outcome : market.outcomes()) {
					if (outcome == null) {
						continue;
					}
					offers.add(new PitcherStrikeoutOffer(
							sportsbookKey,
							sportsbookDisplayName,
							availableAt,
							outcome.description(),
							outcome.name(),
							outcome.point(),
							outcome.price()
					));
				}
			}
		}

		return offers;
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record OddsApiEventSummary(
			String id
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record OddsApiEventDetail(
			String id,
			@JsonProperty("commence_time")
			OffsetDateTime commenceTime,
			@JsonProperty("home_team")
			String homeTeam,
			@JsonProperty("away_team")
			String awayTeam,
			List<OddsApiBookmaker> bookmakers
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record OddsApiBookmaker(
			String key,
			String title,
			@JsonProperty("last_update")
			OffsetDateTime lastUpdate,
			List<OddsApiMarket> markets
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record OddsApiMarket(
			String key,
			List<OddsApiOutcome> outcomes
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record OddsApiOutcome(
			String name,
			String description,
			BigDecimal point,
			Integer price
	) {
	}
}
