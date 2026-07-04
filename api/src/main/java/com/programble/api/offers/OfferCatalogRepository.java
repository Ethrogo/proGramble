package com.programble.api.offers;

import java.util.List;
import java.util.Optional;

import com.programble.api.offers.OfferCatalogService.PlayerDescriptor;
import com.programble.api.offers.OfferCatalogService.ProgrambleOffer;

public interface OfferCatalogRepository {

	Optional<PlayerDescriptor> findPlayer(long playerId);

	List<ProgrambleOffer> findOffersForEvent(long eventId, Long playerId, String sportsbookKey, String marketTypeKey);

	List<ProgrambleOffer> findOffersForPlayer(long playerId, String sportsbookKey, String marketTypeKey);
}
