import { render, screen } from "@solidjs/testing-library";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("ContextBench workbench", () => {
  it("renders the retrieval workbench and its four method lanes", () => {
    render(() => <App />);

    expect(screen.getByRole("heading", { name: /query debugger/i })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: /query/i })).toHaveValue(
      "What happens when a Raft follower falls behind?",
    );
    expect(screen.getByText("Vector search")).toBeInTheDocument();
    expect(screen.getByText("BM25 lexical")).toBeInTheDocument();
    expect(screen.getByText("Hybrid RRF")).toBeInTheDocument();
    expect(screen.getByText("Reranked")).toBeInTheDocument();
  });
});
