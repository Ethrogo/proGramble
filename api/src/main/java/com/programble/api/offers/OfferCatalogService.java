package com.programble.api.offers;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

import org.springframework.stereotype.Service;

@Service
public class OfferCatalogService {

	private final OfferCatalogRepository offerCatalogRepository;

	public OfferCatalogService(OfferCatalogRepository offerCatalogRepository) {
		this.offerCatalogRepository = offerCatalogRepository;
	}

	public Optional<PlayerDescriptor> findPlayer(long playerId) {
		return this.offerCatalogRepository.findPlayer(playerId);
	}

	public List<ProgrambleOffer> findOffersForEvent(long eventId, Long playerId, String sportsbookKey, String marketTypeKey) {
		return this.offerCatalogRepository.findOffersForEvent(
				eventId,
				playerId,
				normalize(sportsbookKey),
				normalize(marketTypeKey)
		);
	}

	public List<ProgrambleOffer> findOffersForPlayer(long playerId, String sportsbookKey, String marketTypeKey) {
		return this.offerCatalogRepository.findOffersForPlayer(
				playerId,
				normalize(sportsbookKey),
				normalize(marketTypeKey)
		);
	}

	private static String normalize(String value) {
		if (value == null) {
			return null;
		}

		var normalized = value.trim().toLowerCase(Locale.US);
		return normalized.isEmpty() ? null : normalized;
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

	public record EventDescriptor(
			long id,
			String slug,
			String externalRef,
			String eventType,
			String status,
			OffsetDateTime scheduledStart,
			boolean startTimeConfirmed,
			SportDescriptor sport,
			CompetitionDescriptor competition,
			VenueDescriptor venue
	) {
	}

	public record PlayerDescriptor(
			long id,
			String slug,
			String displayName,
			SportDescriptor sport
	) {
	}

	public record SportsbookDescriptor(
			long id,
			String code,
			String slug,
			String displayName,
			String regionCode
	) {
	}

	public record MarketDescriptor(
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

	public record OfferParticipantDescriptor(
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

	public record ProgrambleOffer(
			long id,
			EventDescriptor event,
			SportsbookDescriptor sportsbook,
			MarketDescriptor market,
			OfferParticipantDescriptor participant,
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
}
