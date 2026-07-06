package com.programble.api.oddsrefresh;

import java.text.Normalizer;
import java.util.Locale;
import java.util.Map;

final class MlbTeamMappings {

	private static final Map<String, String> TEAM_NAME_TO_ABBR = Map.ofEntries(
			Map.entry("Arizona Diamondbacks", "ARI"),
			Map.entry("Atlanta Braves", "ATL"),
			Map.entry("Athletics", "ATH"),
			Map.entry("Baltimore Orioles", "BAL"),
			Map.entry("Boston Red Sox", "BOS"),
			Map.entry("Chicago Cubs", "CHC"),
			Map.entry("Chicago White Sox", "CHW"),
			Map.entry("Cincinnati Reds", "CIN"),
			Map.entry("Cleveland Guardians", "CLE"),
			Map.entry("Colorado Rockies", "COL"),
			Map.entry("Detroit Tigers", "DET"),
			Map.entry("Houston Astros", "HOU"),
			Map.entry("Kansas City Royals", "KC"),
			Map.entry("Los Angeles Angels", "LAA"),
			Map.entry("Los Angeles Dodgers", "LAD"),
			Map.entry("Miami Marlins", "MIA"),
			Map.entry("Milwaukee Brewers", "MIL"),
			Map.entry("Minnesota Twins", "MIN"),
			Map.entry("New York Mets", "NYM"),
			Map.entry("New York Yankees", "NYY"),
			Map.entry("Philadelphia Phillies", "PHI"),
			Map.entry("Pittsburgh Pirates", "PIT"),
			Map.entry("San Diego Padres", "SD"),
			Map.entry("San Francisco Giants", "SF"),
			Map.entry("Seattle Mariners", "SEA"),
			Map.entry("St. Louis Cardinals", "STL"),
			Map.entry("Tampa Bay Rays", "TB"),
			Map.entry("Texas Rangers", "TEX"),
			Map.entry("Toronto Blue Jays", "TOR"),
			Map.entry("Washington Nationals", "WSH")
	);

	private static final Map<String, String> TEAM_CODE_ALIASES = Map.ofEntries(
			Map.entry("ATH", "ATH"),
			Map.entry("OAK", "ATH"),
			Map.entry("KC", "KC"),
			Map.entry("KCR", "KC"),
			Map.entry("SD", "SD"),
			Map.entry("SDP", "SD"),
			Map.entry("SF", "SF"),
			Map.entry("SFG", "SF"),
			Map.entry("TB", "TB"),
			Map.entry("TBR", "TB"),
			Map.entry("WSH", "WSH"),
			Map.entry("WSN", "WSH")
	);

	private MlbTeamMappings() {
	}

	static String canonicalCode(String teamCodeOrName) {
		if (teamCodeOrName == null || teamCodeOrName.isBlank()) {
			return "";
		}

		String direct = TEAM_NAME_TO_ABBR.get(teamCodeOrName);
		if (direct != null) {
			return direct;
		}

		String normalizedCode = normalizeCode(teamCodeOrName);
		return TEAM_CODE_ALIASES.getOrDefault(normalizedCode, normalizedCode);
	}

	static String slugify(String value) {
		return Normalizer.normalize(value, Normalizer.Form.NFD)
				.replaceAll("\\p{M}", "")
				.toLowerCase(Locale.US)
				.replaceAll("[^a-z0-9]+", "-")
				.replaceAll("(^-|-$)", "");
	}

	private static String normalizeCode(String value) {
		return value.trim().toUpperCase(Locale.US);
	}
}
