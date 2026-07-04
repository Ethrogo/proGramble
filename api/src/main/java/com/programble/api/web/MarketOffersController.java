package com.programble.api.web;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.List;

import com.programble.api.events.EventCatalogService;
import com.programble.api.events.EventCatalogService.ProgrambleEvent;
import com.programble.api.offers.OfferCatalogService;
import com.programble.api.offers.OfferCatalogService.PlayerDescriptor;
import com.programble.api.offers.OfferCatalogService.ProgrambleOffer;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import static org.springframework.http.HttpStatus.NOT_FOUND;

@RestController
@RequestMapping("${programble.api.base-path}")
public class MarketOffersController {

	private final EventCatalogService eventCatalogService;
	private final OfferCatalogService offerCatalogService;

	public MarketOffersController(EventCatalogService eventCatalogService, OfferCatalogService offerCatalogService) {
		this.eventCatalogService = eventCatalogService;
		this.offerCatalogService = offerCatalogService;
	}

	@GetMapping("/events/{eventId}/offers")
	public EventOffersResponse eventOffers(
			@PathVariable long eventId,
			@RequestParam(required = false) Long playerId,
			@RequestParam(required = false) String sportsbook,
			@RequestParam(required = false) String marketType
	) {
		ProgrambleEvent event = this.eventCatalogService.findEvent(eventId)
				.orElseThrow(() -> new ResponseStatusException(NOT_FOUND, "Unknown event id: " + eventId));
		List<ProgrambleOffer> offers = this.offerCatalogService.findOffersForEvent(eventId, playerId, sportsbook, marketType);

		return new EventOffersResponse(
				toEventResponse(event),
				new OfferFiltersResponse(playerId, sportsbook, marketType),
				offers.size(),
				offers.stream().map(MarketOffersController::toOfferResponse).toList()
		);
	}

	@GetMapping("/players/{playerId}/offers")
	public PlayerOffersResponse playerOffers(
			@PathVariable long playerId,
			@RequestParam(required = false) String sportsbook,
			@RequestParam(required = false) String marketType
	) {
		PlayerDescriptor player = this.offerCatalogService.findPlayer(playerId)
				.orElseThrow(() -> new ResponseStatusException(NOT_FOUND, "Unknown player id: " + playerId));
		List<ProgrambleOffer> offers = this.offerCatalogService.findOffersForPlayer(playerId, sportsbook, marketType);

		return new PlayerOffersResponse(
				toPlayerResponse(player),
				new OfferFiltersResponse(null, sportsbook, marketType),
				offers.size(),
				offers.stream().map(MarketOffersController::toOfferResponse).toList()
		);
	}

	private static OfferResponse toOfferResponse(ProgrambleOffer offer) {
		return new OfferResponse(
				offer.id(),
				toEventResponse(offer.event()),
				new OfferSportsbookResponse(
						offer.sportsbook().id(),
						offer.sportsbook().code(),
						offer.sportsbook().slug(),
						offer.sportsbook().displayName(),
						offer.sportsbook().regionCode()
				),
				new OfferMarketResponse(
						offer.market().id(),
						offer.market().code(),
						offer.market().slug(),
						offer.market().displayName(),
						offer.market().marketScope(),
						offer.market().statType(),
						offer.market().periodType(),
						offer.market().allowsOverUnder(),
						offer.market().allowsBinaryOutcome()
				),
				offer.participant() == null ? null : new OfferParticipantResponse(
						offer.participant().eventParticipantId(),
						offer.participant().teamId(),
						offer.participant().playerId(),
						offer.participant().type(),
						offer.participant().roleCode(),
						offer.participant().displayName(),
						offer.participant().shortName(),
						offer.participant().sortOrder(),
						offer.participant().isHome(),
						offer.participant().isAway()
				),
				offer.lineValue(),
				offer.priceAmerican(),
				offer.priceDecimal(),
				offer.selectionLabel(),
				offer.sideCode(),
				offer.outcomeType(),
				offer.availableAt(),
				offer.isLive()
		);
	}

	private static OfferEventResponse toEventResponse(ProgrambleEvent event) {
		return new OfferEventResponse(
				event.id(),
				event.slug(),
				event.externalRef(),
				event.eventType(),
				event.status(),
				event.scheduledStart(),
				event.startTimeConfirmed(),
				new OfferSportResponse(event.sport().code(), event.sport().slug(), event.sport().name()),
				new OfferCompetitionResponse(
						event.competition().code(),
						event.competition().slug(),
						event.competition().name(),
						event.competition().competitionType()
				),
				new OfferVenueResponse(
						event.venue().name(),
						event.venue().city(),
						event.venue().countryCode()
				)
		);
	}

	private static OfferEventResponse toEventResponse(OfferCatalogService.EventDescriptor event) {
		return new OfferEventResponse(
				event.id(),
				event.slug(),
				event.externalRef(),
				event.eventType(),
				event.status(),
				event.scheduledStart(),
				event.startTimeConfirmed(),
				new OfferSportResponse(event.sport().code(), event.sport().slug(), event.sport().name()),
				new OfferCompetitionResponse(
						event.competition().code(),
						event.competition().slug(),
						event.competition().name(),
						event.competition().competitionType()
				),
				new OfferVenueResponse(
						event.venue().name(),
						event.venue().city(),
						event.venue().countryCode()
				)
		);
	}

	private static OfferPlayerResponse toPlayerResponse(PlayerDescriptor player) {
		return new OfferPlayerResponse(
				player.id(),
				player.slug(),
				player.displayName(),
				new OfferSportResponse(player.sport().code(), player.sport().slug(), player.sport().name())
		);
	}
}

record EventOffersResponse(
		OfferEventResponse event,
		OfferFiltersResponse filters,
		int count,
		List<OfferResponse> offers
) {
}

record PlayerOffersResponse(
		OfferPlayerResponse player,
		OfferFiltersResponse filters,
		int count,
		List<OfferResponse> offers
) {
}

record OfferFiltersResponse(
		Long playerId,
		String sportsbook,
		String marketType
) {
}

record OfferResponse(
		long id,
		OfferEventResponse event,
		OfferSportsbookResponse sportsbook,
		OfferMarketResponse market,
		OfferParticipantResponse participant,
		BigDecimal lineValue,
		Integer priceAmerican,
		BigDecimal priceDecimal,
		String selectionLabel,
		String sideCode,
		String outcomeType,
		OffsetDateTime availableAt,
		boolean isLive
) {
}

record OfferEventResponse(
		long id,
		String slug,
		String externalRef,
		String eventType,
		String status,
		OffsetDateTime scheduledStart,
		boolean startTimeConfirmed,
		OfferSportResponse sport,
		OfferCompetitionResponse competition,
		OfferVenueResponse venue
) {
}

record OfferPlayerResponse(
		long id,
		String slug,
		String displayName,
		OfferSportResponse sport
) {
}

record OfferSportsbookResponse(
		long id,
		String code,
		String slug,
		String displayName,
		String regionCode
) {
}

record OfferMarketResponse(
		long id,
		String code,
		String slug,
		String displayName,
		String marketScope,
		String statType,
		String periodType,
		boolean allowsOverUnder,
		boolean allowsBinaryOutcome
) {
}

record OfferParticipantResponse(
		long eventParticipantId,
		Long teamId,
		Long playerId,
		String type,
		String roleCode,
		String displayName,
		String shortName,
		Integer sortOrder,
		Boolean isHome,
		Boolean isAway
) {
}

record OfferSportResponse(
		String code,
		String slug,
		String name
) {
}

record OfferCompetitionResponse(
		String code,
		String slug,
		String name,
		String competitionType
) {
}

record OfferVenueResponse(
		String name,
		String city,
		String countryCode
) {
}
