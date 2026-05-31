import type { ReactNode } from "react";

export function SectionCard({
  title,
  children
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <article className="card">
      <h3>{title}</h3>
      {children}
    </article>
  );
}
