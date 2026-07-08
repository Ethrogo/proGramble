package com.programble.api.oddsrefresh;

import java.io.IOException;
import java.io.Reader;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

@Service
public class MlbTrackedOfferBackfillService {

	private static final Logger log = LoggerFactory.getLogger(MlbTrackedOfferBackfillService.class);

	private static final String MARKET_KEY = "pitcher_strikeouts";
	private static final Resource HISTORICAL_LINES_RESOURCE = new ClassPathResource("bootstrap/mlb/historical_lines.csv");
	private static final Resource SHADOW_PREDICTIONS_RESOURCE = new ClassPathResource("bootstrap/mlb/pitcher_k_shadow_predictions.csv");
	private static final Resource OFFICIAL_PICKS_HISTORY_RESOURCE = new ClassPathResource("bootstrap/mlb/official_picks_history.csv");
	private static final CSVFormat CSV = CSVFormat.DEFAULT.builder()
			.setHeader()
			.setSkipHeaderRecord(true)
			.setIgnoreEmptyLines(true)
			.setTrim(true)
			.build();

	private final OddsRefreshRepository repository;
	private volatile TrackedOfferDataset historicalDataset;
	private volatile TrackedOfferDataset upcomingDataset;

	public MlbTrackedOfferBackfillService(OddsRefreshRepository repository) {
		this.repository = repository;
	}

	public TrackedOfferDataset loadTrackedOffers() {
		return loadDataset(TrackedOfferWindow.HISTORICAL_ONLY);
	}

	public TrackedOfferDataset loadUpcomingTrackedOffers() {
		return loadDataset(TrackedOfferWindow.TODAY_AND_FUTURE);
	}

	public TrackedOfferBackfillSummary backfill(TrackedOfferDataset dataset) {
		if (dataset.offers().isEmpty()) {
			return new TrackedOfferBackfillSummary(0, 0, 0, 0);
		}

		int offersImported = 0;
		int offersSkipped = 0;
		Set<String> sportsbookKeysTouched = new LinkedHashSet<>();
		Set<String> playerRefsTouched = new LinkedHashSet<>();
		Long marketId = null;

		for (TrackedOfferSeed offer : dataset.offers()) {
			Optional<OddsRefreshRepository.EventMatch> eventMatch = this.repository.findMlbEventByTeamCodes(
					offer.gameDate(),
					offer.teamCode(),
					offer.opponentCode()
			);

			if (eventMatch.isEmpty()) {
				offersSkipped++;
				continue;
			}

			OddsRefreshRepository.EventMatch event = eventMatch.get();
			if (marketId == null) {
				marketId = this.repository.upsertPitcherStrikeoutsMarket(event.sportId());
			}

			long sportsbookId = this.repository.upsertSportsbook(
					new OddsRefreshRepository.SportsbookUpsert(
							offer.sportsbookKey().toUpperCase(Locale.US),
							offer.sportsbookKey(),
							offer.sportsbookDisplayName(),
							"US"
					)
			);
			sportsbookKeysTouched.add(offer.sportsbookKey());

			long playerId = this.repository.upsertPlayer(
					new OddsRefreshRepository.PlayerUpsert(
							event.sportId(),
							offer.playerExternalRef(),
							offer.playerDisplayName()
					)
			);
			playerRefsTouched.add(offer.playerExternalRef() == null ? offer.playerDisplayName() : offer.playerExternalRef());

			boolean home = offer.teamCode().equalsIgnoreCase(event.homeTeamCode());
			long participantId = this.repository.ensurePlayerEventParticipant(
					new OddsRefreshRepository.PlayerEventParticipantUpsert(
							event.eventId(),
							playerId,
							home ? "STARTING_PITCHER_HOME" : "STARTING_PITCHER_AWAY",
							home,
							!home,
							home ? 11 : 10
					)
			);

			this.repository.upsertOffer(
					new OddsRefreshRepository.OfferUpsert(
							sportsbookId,
							event.eventId(),
							marketId,
							participantId,
							offer.lineValue(),
							offer.americanPrice(),
							PitcherStrikeoutOfferSupport.toDecimalPrice(offer.americanPrice()),
							PitcherStrikeoutOfferSupport.buildSelectionLabel(
									offer.playerDisplayName(),
									offer.sideCode(),
									offer.lineValue()
							),
							offer.sideCode().toUpperCase(Locale.US),
							"PROP",
							offer.availableAt() == null ? event.scheduledStart() : offer.availableAt(),
							false,
							offer.sourceOfferId()
					)
			);
			offersImported++;
		}

		return new TrackedOfferBackfillSummary(
				offersImported,
				offersSkipped,
				sportsbookKeysTouched.size(),
				playerRefsTouched.size()
		);
	}

	private TrackedOfferDataset loadDataset(TrackedOfferWindow window) {
		TrackedOfferDataset cached = window == TrackedOfferWindow.HISTORICAL_ONLY
				? this.historicalDataset
				: this.upcomingDataset;
		if (cached != null) {
			return cached;
		}

		synchronized (this) {
			if (window == TrackedOfferWindow.HISTORICAL_ONLY) {
				if (this.historicalDataset == null) {
					this.historicalDataset = parseTrackedOffers(window);
				}
				return this.historicalDataset;
			}

			if (this.upcomingDataset == null) {
				this.upcomingDataset = parseTrackedOffers(window);
			}
			return this.upcomingDataset;
		}
	}

	private TrackedOfferDataset parseTrackedOffers(TrackedOfferWindow window) {
		try {
			Map<String, HistoricalMetadata> metadataByKey = new LinkedHashMap<>();
			Map<String, TrackedOfferSeed> offersBySourceId = new LinkedHashMap<>();
			LocalDate today = LocalDate.now(MlbTime.EASTERN_ZONE);

			int shadowRows = 0;
			for (CSVRecord record : readCsv(SHADOW_PREDICTIONS_RESOURCE)) {
				Optional<TrackedOfferSeed> offer = shadowOffer(record, today, window);
				if (offer.isEmpty()) {
					continue;
				}
				shadowRows++;
				TrackedOfferSeed seed = offer.get();
				offersBySourceId.put(seed.sourceOfferId(), seed);
				metadataByKey.put(metadataKey(
						seed.gameDate(),
						seed.trackedEventId(),
						seed.playerExternalRef(),
						normalizeName(recordValue(record, "participant_name_norm")),
						seed.sportsbookKey(),
						seed.sideCode(),
						seed.lineValue()
				), new HistoricalMetadata(seed.teamCode(), seed.opponentCode(), seed.playerExternalRef(), seed.playerDisplayName()));
			}

			int officialRows = 0;
			for (CSVRecord record : readCsv(OFFICIAL_PICKS_HISTORY_RESOURCE)) {
				Optional<TrackedOfferSeed> offer = officialPickOffer(record, today, window);
				if (offer.isEmpty()) {
					continue;
				}
				officialRows++;
				TrackedOfferSeed seed = offer.get();
				offersBySourceId.putIfAbsent(seed.sourceOfferId(), seed);
				metadataByKey.putIfAbsent(
						metadataKey(
								seed.gameDate(),
								seed.trackedEventId(),
								seed.playerExternalRef(),
								normalizeName(seed.playerDisplayName()),
								seed.sportsbookKey(),
								seed.sideCode(),
								seed.lineValue()
						),
						new HistoricalMetadata(seed.teamCode(), seed.opponentCode(), seed.playerExternalRef(), seed.playerDisplayName())
				);
			}

			int historicalRows = 0;
			for (CSVRecord record : readCsv(HISTORICAL_LINES_RESOURCE)) {
				Optional<TrackedOfferSeed> offer = historicalOffer(record, metadataByKey, today, window);
				if (offer.isEmpty()) {
					continue;
				}
				historicalRows++;
				TrackedOfferSeed seed = offer.get();
				offersBySourceId.putIfAbsent(seed.sourceOfferId(), seed);
			}

			List<TrackedOfferSeed> offers = List.copyOf(offersBySourceId.values());
			Set<LocalDate> dates = offers.stream().map(TrackedOfferSeed::gameDate).collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));

			return new TrackedOfferDataset(
					offers,
					Set.copyOf(dates),
					shadowRows,
					historicalRows,
					officialRows
			);
		}
		catch (IOException exception) {
			throw new IllegalStateException("Failed to load tracked MLB backfill resources.", exception);
		}
	}

	private Optional<TrackedOfferSeed> shadowOffer(CSVRecord record, LocalDate today, TrackedOfferWindow window) {
		if (!"MLB".equalsIgnoreCase(recordValue(record, "sport"))
				|| !MARKET_KEY.equalsIgnoreCase(recordValue(record, "market_key"))) {
			return Optional.empty();
		}

		LocalDate gameDate = parseLocalDate(recordValue(record, "game_date"));
		if (gameDate == null || !window.includes(gameDate, today)) {
			return Optional.empty();
		}

		String teamCode = MlbTeamMappings.canonicalCode(recordValue(record, "team"));
		String opponentCode = MlbTeamMappings.canonicalCode(recordValue(record, "opponent"));
		String sportsbookKey = normalizeCode(recordValue(record, "bookmaker_key"));
		String sideCode = PitcherStrikeoutOfferSupport.normalizeSide(recordValue(record, "side"));
		BigDecimal lineValue = parseBigDecimal(recordValue(record, "line"));
		Integer americanPrice = parseInteger(recordValue(record, "price"));
		String trackedEventId = recordValue(record, "event_id");
		String playerExternalRef = normalizePlayerRef(recordValue(record, "participant_join_key"), recordValue(record, "participant_source_id"), recordValue(record, "participant_source_id_type"));

		if (teamCode.isBlank()
				|| opponentCode.isBlank()
				|| sportsbookKey.isBlank()
				|| trackedEventId.isBlank()
				|| lineValue == null
				|| americanPrice == null
				|| !PitcherStrikeoutOfferSupport.isSupportedSide(sideCode)) {
			return Optional.empty();
		}

		String playerName = toDisplayName(recordValue(record, "player_name"));
		return Optional.of(new TrackedOfferSeed(
				gameDate,
				trackedEventId,
				teamCode,
				opponentCode,
				playerExternalRef,
				playerName,
				sportsbookKey,
				recordValue(record, "book"),
				sideCode,
				lineValue,
				americanPrice,
				null,
				buildTrackedSourceOfferId(gameDate, trackedEventId, playerExternalRef, normalizeName(playerName), sportsbookKey, sideCode, lineValue)
		));
	}

	private Optional<TrackedOfferSeed> officialPickOffer(CSVRecord record, LocalDate today, TrackedOfferWindow window) {
		String marketKey = recordValue(record, "market_key");
		String sport = recordValue(record, "sport");
		LocalDate gameDate = parseLocalDate(recordValue(record, "game_date"));

		if (!"MLB".equalsIgnoreCase(sport)
				|| !MARKET_KEY.equalsIgnoreCase(marketKey)
				|| gameDate == null
				|| !window.includes(gameDate, today)) {
			return Optional.empty();
		}

		String teamCode = MlbTeamMappings.canonicalCode(recordValue(record, "team"));
		String opponentCode = MlbTeamMappings.canonicalCode(recordValue(record, "opponent"));
		String trackedEventId = recordValue(record, "event_id");
		String sportsbookKey = normalizeCode(nonBlank(recordValue(record, "bookmaker_key"), recordValue(record, "book")));
		String sideCode = PitcherStrikeoutOfferSupport.normalizeSide(recordValue(record, "pick_side"));
		BigDecimal lineValue = parseBigDecimal(recordValue(record, "line"));
		String playerExternalRef = normalizePlayerRef(recordValue(record, "participant_join_key"), recordValue(record, "participant_source_id"), recordValue(record, "participant_source_id_type"));
		String playerName = toDisplayName(recordValue(record, "player_name"));
		Integer americanPrice = parseInteger(nonBlank(recordValue(record, "price"), recordValue(record, "odds")));

		if (teamCode.isBlank()
				|| opponentCode.isBlank()
				|| trackedEventId.isBlank()
				|| sportsbookKey.isBlank()
				|| lineValue == null
				|| americanPrice == null
				|| !PitcherStrikeoutOfferSupport.isSupportedSide(sideCode)) {
			return Optional.empty();
		}

		return Optional.of(new TrackedOfferSeed(
				gameDate,
				trackedEventId,
				teamCode,
				opponentCode,
				playerExternalRef,
				playerName,
				sportsbookKey,
				nonBlank(recordValue(record, "book"), sportsbookKey),
				sideCode,
				lineValue,
				americanPrice,
				null,
				buildTrackedSourceOfferId(gameDate, trackedEventId, playerExternalRef, normalizeName(playerName), sportsbookKey, sideCode, lineValue)
		));
	}

	private Optional<TrackedOfferSeed> historicalOffer(
			CSVRecord record,
			Map<String, HistoricalMetadata> metadataByKey,
			LocalDate today,
			TrackedOfferWindow window
	) {
		if (!"MLB".equalsIgnoreCase(recordValue(record, "sport"))
				|| !MARKET_KEY.equalsIgnoreCase(recordValue(record, "market_key"))) {
			return Optional.empty();
		}

		LocalDate gameDate = parseLocalDate(recordValue(record, "game_date"));
		if (gameDate == null || !window.includes(gameDate, today)) {
			return Optional.empty();
		}

		String sportsbookKey = normalizeCode(recordValue(record, "bookmaker_key"));
		String sideCode = PitcherStrikeoutOfferSupport.normalizeSide(nonBlank(recordValue(record, "side_norm"), recordValue(record, "side")));
		BigDecimal lineValue = parseBigDecimal(recordValue(record, "line"));
		Integer americanPrice = parseInteger(recordValue(record, "price"));
		String trackedEventId = recordValue(record, "event_id");
		String playerExternalRef = normalizePlayerRef(
				nonBlank(recordValue(record, "participant_source_key"), recordValue(record, "participant_join_key")),
				recordValue(record, "participant_source_id"),
				recordValue(record, "participant_source_id_type")
		);
		String playerName = toDisplayName(nonBlank(recordValue(record, "participant_name"), recordValue(record, "player_name")));

		if (sportsbookKey.isBlank()
				|| trackedEventId.isBlank()
				|| lineValue == null
				|| americanPrice == null
				|| !PitcherStrikeoutOfferSupport.isSupportedSide(sideCode)) {
			return Optional.empty();
		}

		HistoricalMetadata metadata = metadataByKey.get(
				metadataKey(
						gameDate,
						trackedEventId,
						playerExternalRef,
						normalizeName(nonBlank(recordValue(record, "participant_name_norm"), recordValue(record, "player_name_norm"))),
						sportsbookKey,
						sideCode,
						lineValue
				)
		);
		if (metadata == null) {
			return Optional.empty();
		}

		String resolvedPlayerRef = metadata.playerExternalRef() == null ? playerExternalRef : metadata.playerExternalRef();
		String resolvedPlayerName = metadata.playerDisplayName() == null || metadata.playerDisplayName().isBlank()
				? playerName
				: metadata.playerDisplayName();

		return Optional.of(new TrackedOfferSeed(
				gameDate,
				trackedEventId,
				metadata.teamCode(),
				metadata.opponentCode(),
				resolvedPlayerRef,
				resolvedPlayerName,
				sportsbookKey,
				nonBlank(recordValue(record, "bookmaker"), sportsbookKey),
				sideCode,
				lineValue,
				americanPrice,
				parseOffsetDateTime(nonBlank(recordValue(record, "pulled_at"), recordValue(record, "commence_time"))),
				buildTrackedSourceOfferId(gameDate, trackedEventId, resolvedPlayerRef, normalizeName(resolvedPlayerName), sportsbookKey, sideCode, lineValue)
		));
	}

	private static Iterable<CSVRecord> readCsv(Resource resource) throws IOException {
		if (!resource.exists()) {
			log.warn("Tracked MLB resource is missing and will be skipped: {}", resource);
			return List.of();
		}

		try (Reader reader = new java.io.InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8);
		     CSVParser parser = CSV.parse(reader)) {
			return parser.getRecords();
		}
	}

	private static String metadataKey(
			LocalDate gameDate,
			String trackedEventId,
			String playerExternalRef,
			String playerNameNorm,
			String sportsbookKey,
			String sideCode,
			BigDecimal lineValue
	) {
		String participantKey = playerExternalRef == null || playerExternalRef.isBlank() ? playerNameNorm : playerExternalRef;
		return String.join(
				"|",
				gameDate.toString(),
				blankSafe(trackedEventId),
				blankSafe(participantKey),
				normalizeCode(sportsbookKey),
				PitcherStrikeoutOfferSupport.normalizeSide(sideCode),
				lineValue.stripTrailingZeros().toPlainString()
		);
	}

	private static String buildTrackedSourceOfferId(
			LocalDate gameDate,
			String trackedEventId,
			String playerExternalRef,
			String playerNameNorm,
			String sportsbookKey,
			String sideCode,
			BigDecimal lineValue
	) {
		String participantToken = playerExternalRef == null || playerExternalRef.isBlank() ? playerNameNorm : playerExternalRef;
		String raw = String.join(
				"|",
				"tracked",
				MARKET_KEY,
				gameDate.toString(),
				blankSafe(trackedEventId),
				blankSafe(participantToken),
				normalizeCode(sportsbookKey),
				PitcherStrikeoutOfferSupport.normalizeSide(sideCode),
				lineValue.stripTrailingZeros().toPlainString()
		);

		if (raw.length() <= 128) {
			return raw;
		}

		return "tracked|h|" + sha256(raw);
	}

	private static String sha256(String value) {
		try {
			MessageDigest digest = MessageDigest.getInstance("SHA-256");
			byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
			StringBuilder builder = new StringBuilder();
			for (byte current : bytes) {
				builder.append(String.format("%02x", current));
			}
			return builder.substring(0, 32);
		}
		catch (NoSuchAlgorithmException exception) {
			throw new IllegalStateException("SHA-256 is not available.", exception);
		}
	}

	private static String normalizePlayerRef(String participantJoinKey, String participantSourceId, String participantSourceIdType) {
		String joinKey = blankToNull(participantJoinKey);
		if (joinKey != null && joinKey.startsWith("mlbam_player:")) {
			return joinKey;
		}

		String sourceId = blankToNull(participantSourceId);
		String sourceType = blankToNull(participantSourceIdType);
		if (sourceId == null || sourceType == null) {
			return null;
		}
		if ("mlbam_player".equalsIgnoreCase(sourceType)) {
			return "mlbam_player:" + sourceId.replace("mlbam_player:", "");
		}
		return sourceType.toLowerCase(Locale.US) + ":" + sourceId;
	}

	private static String toDisplayName(String value) {
		String normalized = blankToNull(value);
		if (normalized == null) {
			return "";
		}

		StringBuilder out = new StringBuilder(normalized.length());
		boolean capitalizeNext = true;
		for (char current : normalized.toLowerCase(Locale.US).toCharArray()) {
			if (capitalizeNext && Character.isLetter(current)) {
				out.append(Character.toUpperCase(current));
				capitalizeNext = false;
			}
			else {
				out.append(current);
				capitalizeNext = current == ' ' || current == '-' || current == '\'';
			}
		}
		return out.toString();
	}

	private static OffsetDateTime parseOffsetDateTime(String value) {
		String normalized = blankToNull(value);
		if (normalized == null) {
			return null;
		}
		return OffsetDateTime.parse(normalized);
	}

	private static LocalDate parseLocalDate(String value) {
		String normalized = blankToNull(value);
		if (normalized == null) {
			return null;
		}
		return LocalDate.parse(normalized);
	}

	private static Integer parseInteger(String value) {
		String normalized = blankToNull(value);
		if (normalized == null) {
			return null;
		}
		return Integer.parseInt(normalized);
	}

	private static BigDecimal parseBigDecimal(String value) {
		String normalized = blankToNull(value);
		if (normalized == null) {
			return null;
		}
		return new BigDecimal(normalized);
	}

	private static String normalizeName(String value) {
		String normalized = blankToNull(value);
		if (normalized == null) {
			return "";
		}
		return normalized.trim().toLowerCase(Locale.US);
	}

	private static String normalizeCode(String value) {
		String normalized = blankToNull(value);
		return normalized == null ? "" : normalized.toLowerCase(Locale.US);
	}

	private static String recordValue(CSVRecord record, String header) {
		return record.isMapped(header) ? record.get(header) : "";
	}

	private static String nonBlank(String primary, String fallback) {
		return blankToNull(primary) == null ? fallback : primary;
	}

	private static String blankSafe(String value) {
		return value == null ? "" : value;
	}

	private static String blankToNull(String value) {
		return value == null || value.isBlank() ? null : value.trim();
	}

	public record TrackedOfferDataset(
			List<TrackedOfferSeed> offers,
			Set<LocalDate> dates,
			int shadowRowsLoaded,
			int historicalRowsLoaded,
			int officialRowsLoaded
	) {
		public int totalOffers() {
			return this.offers.size();
		}
	}

	public record TrackedOfferBackfillSummary(
			int offersImported,
			int offersSkipped,
			int sportsbooksTouched,
			int playersTouched
	) {
	}

	record TrackedOfferSeed(
			LocalDate gameDate,
			String trackedEventId,
			String teamCode,
			String opponentCode,
			String playerExternalRef,
			String playerDisplayName,
			String sportsbookKey,
			String sportsbookDisplayName,
			String sideCode,
			BigDecimal lineValue,
			Integer americanPrice,
			OffsetDateTime availableAt,
			String sourceOfferId
	) {
	}

	private record HistoricalMetadata(
			String teamCode,
			String opponentCode,
			String playerExternalRef,
			String playerDisplayName
	) {
	}

	private enum TrackedOfferWindow {
		HISTORICAL_ONLY {
			@Override
			boolean includes(LocalDate gameDate, LocalDate today) {
				return gameDate.isBefore(today);
			}
		},
		TODAY_AND_FUTURE {
			@Override
			boolean includes(LocalDate gameDate, LocalDate today) {
				return !gameDate.isBefore(today);
			}
		};

		abstract boolean includes(LocalDate gameDate, LocalDate today);
	}
}
