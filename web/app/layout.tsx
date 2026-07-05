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
    "ProGramble helps sports fans find featured slates, popular props, and game-day storylines faster across the sports they follow most."
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
