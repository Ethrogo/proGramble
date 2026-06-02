package com.programble.api.config;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest(properties = {
		"SERVER_PORT=9090",
		"PROGRAMBLE_DB_URL=jdbc:postgresql://db.internal:5432/programble_staging",
		"PROGRAMBLE_DB_USERNAME=programble_app",
		"PROGRAMBLE_DB_PASSWORD=secret"
})
@ActiveProfiles("staging")
class StagingProfilePropertiesTest {

	@Autowired
	private ApiProperties apiProperties;

	@Test
	void stagingProfileBindsRuntimeContract() {
		assertThat(apiProperties.environment()).isEqualTo("staging");
		assertThat(apiProperties.api().basePath()).isEqualTo("/api/v1");
		assertThat(apiProperties.database().url()).isEqualTo("jdbc:postgresql://db.internal:5432/programble_staging");
		assertThat(apiProperties.database().username()).isEqualTo("programble_app");
		assertThat(apiProperties.database().password()).isEqualTo("secret");
	}
}
