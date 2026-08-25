import { render, screen } from "@solidjs/testing-library";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("ContextBench workbench", () => {
  it("renders the local-first empty state without fabricated results", () => {
    render(() => <App />);

    expect(
      screen.getByRole("heading", { name: /query debugger/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /choose a local project first/i }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Vector search")).not.toBeInTheDocument();
    expect(screen.getByText("LOCAL ONLY")).toBeInTheDocument();
  });
});
