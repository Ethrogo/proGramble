package com.programble.api.config;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest(properties = {
		"SERVER_PORT=9090",
		"PROGRAMBLE_DB_URL=jdbc:h2:mem:stagingprops;MODE=PostgreSQL;DATABASE_TO_LOWER=TRUE;DEFAULT_NULL_ORDERING=HIGH;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE;INIT=CREATE DOMAIN IF NOT EXISTS TIMESTAMPTZ AS TIMESTAMP WITH TIME ZONE",
		"PROGRAMBLE_DB_USERNAME=programble_app",
		"PROGRAMBLE_DB_PASSWORD=secret",
		"spring.flyway.locations=classpath:db/migration-h2"
})
@ActiveProfiles("staging")
class StagingProfilePropertiesTest {

	@Autowired
	private ApiProperties apiProperties;

	@Test
	void stagingProfileBindsRuntimeContract() {
		assertThat(apiProperties.environment()).isEqualTo("staging");
		assertThat(apiProperties.api().basePath()).isEqualTo("/api/v1");
		assertThat(apiProperties.database().url()).isEqualTo("jdbc:h2:mem:stagingprops;MODE=PostgreSQL;DATABASE_TO_LOWER=TRUE;DEFAULT_NULL_ORDERING=HIGH;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE;INIT=CREATE DOMAIN IF NOT EXISTS TIMESTAMPTZ AS TIMESTAMP WITH TIME ZONE");
		assertThat(apiProperties.database().username()).isEqualTo("programble_app");
		assertThat(apiProperties.database().password()).isEqualTo("secret");
	}

	@Test
	void stagingProfileDeclaresPostgresDatasourceContract() throws IOException {
		var resource = new ClassPathResource("application-staging.properties");
		var stagingProperties = new String(resource.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

		assertThat(stagingProperties).contains("spring.datasource.url=${PROGRAMBLE_DB_URL:jdbc:postgresql://localhost:5432/programble}");
		assertThat(stagingProperties).contains("spring.datasource.username=${PROGRAMBLE_DB_USERNAME:programble}");
		assertThat(stagingProperties).contains("spring.datasource.password=${PROGRAMBLE_DB_PASSWORD:change-me}");
		assertThat(stagingProperties).contains("spring.flyway.enabled=true");
	}
}
