package com.programble.api.oddsrefresh;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Locale;

final class PitcherStrikeoutOfferSupport {

	private PitcherStrikeoutOfferSupport() {
	}

	static BigDecimal toDecimalPrice(Integer americanPrice) {
		if (americanPrice == null || americanPrice == 0) {
			return null;
		}
		BigDecimal result;
		if (americanPrice > 0) {
			result = BigDecimal.ONE.add(BigDecimal.valueOf(americanPrice).divide(BigDecimal.valueOf(100), 4, RoundingMode.HALF_UP));
		}
		else {
			result = BigDecimal.ONE.add(BigDecimal.valueOf(100).divide(BigDecimal.valueOf(Math.abs(americanPrice)), 4, RoundingMode.HALF_UP));
		}
		return result.setScale(4, RoundingMode.HALF_UP);
	}

	static String buildSelectionLabel(String pitcherName, String side, BigDecimal line) {
		return pitcherName + " " + capitalize(side) + " " + line.stripTrailingZeros().toPlainString() + " Strikeouts";
	}

	static String normalizeSide(String side) {
		return side == null ? "" : side.trim().toLowerCase(Locale.US);
	}

	static boolean isSupportedSide(String side) {
		String normalized = normalizeSide(side);
		return "over".equals(normalized) || "under".equals(normalized);
	}

	private static String capitalize(String value) {
		if (value == null || value.isBlank()) {
			return "";
		}
		return value.substring(0, 1).toUpperCase(Locale.US) + value.substring(1).toLowerCase(Locale.US);
	}
}
