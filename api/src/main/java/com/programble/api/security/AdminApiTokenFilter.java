package com.programble.api.security;

import java.io.IOException;
import java.util.List;

import com.programble.api.config.ApiProperties;
import com.programble.api.config.SecurityProperties;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
public class AdminApiTokenFilter extends OncePerRequestFilter {

	private static final String BEARER_PREFIX = "Bearer ";

	private final String adminJobsPrefix;
	private final SecurityProperties securityProperties;

	public AdminApiTokenFilter(ApiProperties apiProperties, SecurityProperties securityProperties) {
		this.adminJobsPrefix = apiProperties.api().basePath() + "/admin/jobs";
		this.securityProperties = securityProperties;
	}

	@Override
	protected boolean shouldNotFilter(HttpServletRequest request) {
		return !request.getRequestURI().startsWith(this.adminJobsPrefix);
	}

	@Override
	protected void doFilterInternal(
			HttpServletRequest request,
			HttpServletResponse response,
			FilterChain filterChain
	) throws ServletException, IOException {
		String expectedToken = this.securityProperties.adminApiToken();
		if (!StringUtils.hasText(expectedToken)) {
			response.sendError(HttpServletResponse.SC_UNAUTHORIZED);
			return;
		}

		String authorization = request.getHeader("Authorization");
		if (!StringUtils.hasText(authorization) || !authorization.startsWith(BEARER_PREFIX)) {
			response.sendError(HttpServletResponse.SC_UNAUTHORIZED);
			return;
		}

		String providedToken = authorization.substring(BEARER_PREFIX.length()).trim();
		if (!expectedToken.equals(providedToken)) {
			response.sendError(HttpServletResponse.SC_UNAUTHORIZED);
			return;
		}

		var authentication = new UsernamePasswordAuthenticationToken(
				"admin-jobs",
				null,
				List.of(new SimpleGrantedAuthority("ROLE_ADMIN"))
		);
		SecurityContextHolder.getContext().setAuthentication(authentication);

		try {
			filterChain.doFilter(request, response);
		}
		finally {
			SecurityContextHolder.clearContext();
		}
	}
}
