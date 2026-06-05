package com.programble.api.schema;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

class InitialSchemaMigrationTest {

	@Test
	void initialSchemaIncludesPortableMultiSportRelationships() throws IOException {
		var resource = new ClassPathResource("db/migration/V1__initial_schema.sql");
		var schemaSql = new String(resource.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

		assertThat(schemaSql).contains("create table sports");
		assertThat(schemaSql).contains("create table competitions");
		assertThat(schemaSql).contains("create table teams");
		assertThat(schemaSql).contains("create table players");
		assertThat(schemaSql).contains("create table events");
		assertThat(schemaSql).contains("create table sportsbooks");
		assertThat(schemaSql).contains("create table markets");
		assertThat(schemaSql).contains("create table offers");
		assertThat(schemaSql).contains("create table competition_teams");
		assertThat(schemaSql).contains("create table team_players");
		assertThat(schemaSql).contains("create table event_participants");
		assertThat(schemaSql).contains("where competition_id is null");
		assertThat(schemaSql).contains("where competition_id is not null");
	}
}
