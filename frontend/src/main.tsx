import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Fonts are bundled with the app (no requests to Google or anyone else).
import "@fontsource-variable/inter";
import "@fontsource-variable/instrument-sans";
import App from "./App";
import "./styles.css";
import "./brand.css";
import "./lab.css";
import "./dash.css";
import "./arena.css";
import "./tests.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
