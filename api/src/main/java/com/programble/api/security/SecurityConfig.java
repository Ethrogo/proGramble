package com.programble.api.security;

import com.programble.api.config.ApiProperties;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AnonymousAuthenticationFilter;
import org.springframework.security.web.servlet.util.matcher.PathPatternRequestMatcher;
import org.springframework.security.web.util.matcher.OrRequestMatcher;
import org.springframework.security.web.util.matcher.RequestMatcher;

@Configuration
public class SecurityConfig {

	@Bean
	UserDetailsService userDetailsService() {
		return new InMemoryUserDetailsManager();
	}

	@Bean
	SecurityFilterChain securityFilterChain(
			HttpSecurity http,
			ApiProperties apiProperties,
			AdminApiTokenFilter adminApiTokenFilter
	) throws Exception {
		String adminJobsBasePath = apiProperties.api().basePath() + "/admin/jobs";
		PathPatternRequestMatcher.Builder matcherBuilder = PathPatternRequestMatcher.withDefaults();
		RequestMatcher adminJobsMatcher = new OrRequestMatcher(
				matcherBuilder.matcher(adminJobsBasePath),
				matcherBuilder.matcher(adminJobsBasePath + "/**")
		);

		http
				.csrf(AbstractHttpConfigurer::disable)
				.formLogin(AbstractHttpConfigurer::disable)
				.httpBasic(AbstractHttpConfigurer::disable)
				.logout(AbstractHttpConfigurer::disable)
				.sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
				.authorizeHttpRequests(authorize -> authorize
						.requestMatchers(adminJobsMatcher).authenticated()
						.anyRequest().permitAll()
				)
				.exceptionHandling(exceptionHandling -> exceptionHandling
						.authenticationEntryPoint((request, response, authException) ->
								response.sendError(HttpServletResponse.SC_UNAUTHORIZED))
				)
				.addFilterBefore(adminApiTokenFilter, AnonymousAuthenticationFilter.class)
				.anonymous(Customizer.withDefaults());

		return http.build();
	}
}
