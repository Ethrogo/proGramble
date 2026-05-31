import Link from "next/link";

type HeroAction = {
  href: string;
  label: string;
  variant?: "primary" | "secondary";
};

export function PageHero({
  eyebrow,
  title,
  description,
  actions,
  sideTitle,
  sideDescription,
  stats
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: HeroAction[];
  sideTitle: string;
  sideDescription: string;
  stats: Array<{ label: string; value: string }>;
}) {
  return (
    <section className="hero">
      <div className="hero-grid">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p>{description}</p>
          {actions && actions.length > 0 ? (
            <div className="hero-actions">
              {actions.map((action) => (
                <Link
                  key={action.href}
                  href={action.href}
                  className={`button ${action.variant ?? "primary"}`}
                >
                  {action.label}
                </Link>
              ))}
            </div>
          ) : null}
        </div>
        <aside className="hero-side">
          <h2>{sideTitle}</h2>
          <p className="kicker">{sideDescription}</p>
          <div className="stat-row">
            {stats.map((stat) => (
              <div key={stat.label} className="stat-chip">
                <span>{stat.label}</span>
                <strong>{stat.value}</strong>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </section>
  );
}
