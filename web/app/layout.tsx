import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteChrome } from "../components/site-chrome";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "ProGramble",
    template: "%s | ProGramble"
  },
  description:
    "A multi-sport shell for projections, slates, props, and yesterday's model-tracked results."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SiteChrome>{children}</SiteChrome>
      </body>
    </html>
  );
}
