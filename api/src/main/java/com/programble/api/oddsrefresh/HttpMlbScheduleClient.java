package com.programble.api.oddsrefresh;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.programble.api.config.MlbStatsApiProperties;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@Component
public class HttpMlbScheduleClient implements MlbScheduleClient {

	private final MlbStatsApiProperties properties;
	private final RestTemplate restTemplate;

	public HttpMlbScheduleClient(MlbStatsApiProperties properties, RestTemplateBuilder restTemplateBuilder) {
		this.properties = properties;
		this.restTemplate = restTemplateBuilder.build();
	}

	@Override
	public List<MlbScheduledGame> fetchGames(LocalDate date) {
		ResponseEntity<MlbScheduleResponse> response = this.restTemplate.getForEntity(
				UriComponentsBuilder.fromHttpUrl(this.properties.statsApiBaseUrl())
						.path("/schedule")
						.queryParam("sportId", 1)
						.queryParam("date", date)
						.queryParam("hydrate", "probablePitcher,team")
						.toUriString(),
				MlbScheduleResponse.class
		);

		MlbScheduleResponse body = response.getBody();
		if (body == null || body.dates() == null) {
			return List.of();
		}

		List<MlbScheduledGame> games = new ArrayList<>();
		for (ScheduleDateBlock dateBlock : body.dates()) {
			if (dateBlock.games() == null) {
				continue;
			}
			for (ScheduleGame game : dateBlock.games()) {
				if (game.teams() == null || game.teams().home() == null || game.teams().away() == null) {
					continue;
				}
				games.add(new MlbScheduledGame(
						String.valueOf(game.gamePk()),
						game.gameDate(),
						game.gameType(),
						mapStatus(game.status()),
						mapVenue(game.venue()),
						mapTeam(game.teams().home().team()),
						mapTeam(game.teams().away().team()),
						mapPitcher(game.teams().home().probablePitcher()),
						mapPitcher(game.teams().away().probablePitcher())
				));
			}
		}

		return List.copyOf(games);
	}

	private static ScheduledPitcher mapPitcher(ProbablePitcher pitcher) {
		if (pitcher == null || pitcher.id() == null || pitcher.fullName() == null) {
			return null;
		}
		return new ScheduledPitcher(pitcher.id(), pitcher.fullName());
	}

	private static ScheduledTeam mapTeam(ScheduleTeam team) {
		if (team == null || team.id() == null || team.name() == null) {
			return null;
		}
		String abbreviation = team.abbreviation() == null ? MlbTeamMappings.canonicalCode(team.name()) : MlbTeamMappings.canonicalCode(team.abbreviation());
		return new ScheduledTeam(
				team.id(),
				team.name(),
				abbreviation,
				team.locationName()
		);
	}

	private static ScheduledVenue mapVenue(ScheduleVenue venue) {
		if (venue == null) {
			return new ScheduledVenue(null, null, null);
		}
		return new ScheduledVenue(
				venue.name(),
				venue.location() == null ? null : venue.location().city(),
				venue.location() == null ? null : venue.location().countryCode()
		);
	}

	private static String mapStatus(ScheduleStatus status) {
		if (status == null) {
			return "SCHEDULED";
		}
		String abstractState = status.abstractGameState();
		if (abstractState == null) {
			return "SCHEDULED";
		}
		return switch (abstractState.trim().toLowerCase()) {
			case "preview", "scheduled", "pre-game" -> "SCHEDULED";
			case "live", "in progress", "manager challenge", "delayed" -> "IN_PROGRESS";
			case "final", "game over" -> "FINAL";
			default -> abstractState.trim().toUpperCase().replace(' ', '_');
		};
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record MlbScheduleResponse(
			List<ScheduleDateBlock> dates
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleDateBlock(
			List<ScheduleGame> games
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleGame(
			Integer gamePk,
			@JsonProperty("gameDate")
			OffsetDateTime gameDate,
			String gameType,
			ScheduleStatus status,
			ScheduleVenue venue,
			ScheduleTeams teams
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleTeams(
			ScheduleSide home,
			ScheduleSide away
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleSide(
			ScheduleTeam team,
			ProbablePitcher probablePitcher
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleTeam(
			Long id,
			String abbreviation,
			String locationName,
			String name
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleStatus(
			String abstractGameState
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleVenue(
			String name,
			ScheduleVenueLocation location
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ScheduleVenueLocation(
			String city,
			String countryCode
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ProbablePitcher(
			Long id,
			String fullName
	) {
	}
}
