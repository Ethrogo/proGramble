create table sports (
    id bigserial primary key,
    code varchar(32) not null unique,
    slug varchar(64) not null unique,
    name varchar(128) not null,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table competitions (
    id bigserial primary key,
    sport_id bigint not null references sports(id),
    code varchar(64) not null,
    slug varchar(96) not null,
    name varchar(128) not null,
    competition_type varchar(32) not null,
    gender_category varchar(32),
    level_name varchar(64),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint competitions_type_chk
        check (competition_type in ('TEAM', 'INDIVIDUAL', 'MIXED')),
    constraint competitions_code_uq unique (sport_id, code),
    constraint competitions_slug_uq unique (sport_id, slug)
);

create table teams (
    id bigserial primary key,
    sport_id bigint not null references sports(id),
    slug varchar(96) not null,
    code varchar(32),
    external_ref varchar(128),
    short_name varchar(96) not null,
    full_name varchar(160) not null,
    city varchar(96),
    country_code char(2),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint teams_slug_uq unique (sport_id, slug),
    constraint teams_external_ref_uq unique (sport_id, external_ref)
);

create table competition_teams (
    id bigserial primary key,
    competition_id bigint not null references competitions(id) on delete cascade,
    team_id bigint not null references teams(id) on delete cascade,
    external_ref varchar(128),
    joined_at timestamptz,
    left_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint competition_teams_membership_uq unique (competition_id, team_id),
    constraint competition_teams_external_ref_uq unique (competition_id, external_ref),
    constraint competition_teams_dates_chk
        check (left_at is null or joined_at is null or left_at >= joined_at)
);

create table players (
    id bigserial primary key,
    sport_id bigint not null references sports(id),
    slug varchar(128) not null,
    external_ref varchar(128),
    first_name varchar(96),
    last_name varchar(96),
    display_name varchar(160) not null,
    country_code char(2),
    handedness varchar(16),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint players_slug_uq unique (sport_id, slug),
    constraint players_external_ref_uq unique (sport_id, external_ref)
);

create table team_players (
    id bigserial primary key,
    team_id bigint not null references teams(id) on delete cascade,
    player_id bigint not null references players(id) on delete cascade,
    competition_id bigint references competitions(id) on delete cascade,
    roster_status varchar(32),
    squad_number varchar(16),
    position_code varchar(32),
    effective_from date,
    effective_to date,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint team_players_assignment_uq unique (team_id, player_id, competition_id, effective_from),
    constraint team_players_dates_chk
        check (effective_to is null or effective_from is null or effective_to >= effective_from)
);

create table events (
    id bigserial primary key,
    sport_id bigint not null references sports(id),
    competition_id bigint not null references competitions(id),
    slug varchar(128) not null,
    external_ref varchar(128),
    event_type varchar(32) not null,
    status varchar(32) not null,
    season_label varchar(64),
    round_label varchar(64),
    scheduled_start timestamptz not null,
    start_time_confirmed boolean not null default false,
    venue_name varchar(160),
    venue_city varchar(96),
    venue_country_code char(2),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint events_slug_uq unique (competition_id, slug),
    constraint events_external_ref_uq unique (competition_id, external_ref),
    constraint events_type_chk
        check (event_type in ('GAME', 'MATCH', 'ROUND', 'TOURNAMENT', 'OTHER'))
);

create table event_participants (
    id bigserial primary key,
    event_id bigint not null references events(id) on delete cascade,
    team_id bigint references teams(id),
    player_id bigint references players(id),
    role_code varchar(32) not null,
    seed_value integer,
    sort_order integer,
    is_home boolean,
    is_away boolean,
    created_at timestamptz not null default now(),
    constraint event_participants_subject_chk
        check (
            (team_id is not null and player_id is null)
            or (team_id is null and player_id is not null)
        ),
    constraint event_participants_role_uq unique (event_id, role_code, team_id, player_id)
);

create table sportsbooks (
    id bigserial primary key,
    code varchar(64) not null unique,
    slug varchar(96) not null unique,
    display_name varchar(128) not null,
    region_code varchar(32),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table markets (
    id bigserial primary key,
    sport_id bigint not null references sports(id),
    competition_id bigint references competitions(id),
    code varchar(96) not null,
    slug varchar(128) not null,
    display_name varchar(160) not null,
    market_scope varchar(32) not null,
    stat_type varchar(64),
    period_type varchar(32) not null default 'FULL_EVENT',
    allows_over_under boolean not null default false,
    allows_binary_outcome boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint markets_scope_chk
        check (market_scope in ('EVENT', 'TEAM', 'PLAYER'))
);

create table offers (
    id bigserial primary key,
    sportsbook_id bigint not null references sportsbooks(id),
    event_id bigint not null references events(id) on delete cascade,
    market_id bigint not null references markets(id),
    event_participant_id bigint references event_participants(id),
    line_value numeric(10, 3),
    price_american integer,
    price_decimal numeric(10, 4),
    selection_label varchar(128) not null,
    side_code varchar(32),
    outcome_type varchar(32),
    available_at timestamptz not null default now(),
    is_live boolean not null default false,
    source_offer_id varchar(128),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint offers_side_chk
        check (
            side_code is null
            or side_code in ('OVER', 'UNDER', 'YES', 'NO', 'HOME', 'AWAY', 'PLAYER', 'FIELD')
        ),
    constraint offers_outcome_chk
        check (
            outcome_type is null
            or outcome_type in ('LINE', 'MONEYLINE', 'SPREAD', 'TOTAL', 'PROP', 'FINISH_POSITION')
        ),
    constraint offers_price_decimal_chk
        check (price_decimal is null or price_decimal >= 1.0000)
);

create unique index idx_markets_sport_code_global
    on markets(sport_id, code)
    where competition_id is null;

create unique index idx_markets_competition_code_specific
    on markets(competition_id, code)
    where competition_id is not null;

create unique index idx_markets_sport_slug_global
    on markets(sport_id, slug)
    where competition_id is null;

create unique index idx_markets_competition_slug_specific
    on markets(competition_id, slug)
    where competition_id is not null;

create unique index idx_offers_source_offer_id
    on offers(sportsbook_id, source_offer_id)
    where source_offer_id is not null;

create index idx_competitions_sport_id on competitions(sport_id);
create index idx_teams_sport_id on teams(sport_id);
create index idx_competition_teams_team_id on competition_teams(team_id);
create index idx_players_sport_id on players(sport_id);
create index idx_team_players_player_id on team_players(player_id);
create index idx_team_players_competition_id on team_players(competition_id);
create index idx_events_competition_start on events(competition_id, scheduled_start);
create index idx_event_participants_event_id on event_participants(event_id);
create index idx_markets_sport_competition on markets(sport_id, competition_id);
create index idx_offers_event_market_book on offers(event_id, market_id, sportsbook_id);
create index idx_offers_available_at on offers(available_at desc);
