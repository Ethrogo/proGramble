package com.programble.api.oddsrefresh;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
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
				games.add(new MlbScheduledGame(
						String.valueOf(game.gamePk()),
						game.teams().home().team().name(),
						game.teams().away().team().name(),
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
			String name
	) {
	}

	@JsonIgnoreProperties(ignoreUnknown = true)
	private record ProbablePitcher(
			Long id,
			String fullName
	) {
	}
}
