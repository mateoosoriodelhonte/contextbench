import { test } from "@playwright/test";
import { WorkbenchPage } from "./pages/WorkbenchPage";

test("runs the complete local retrieval evaluation flow", async ({ page }) => {
  const workbench = new WorkbenchPage(page);
  await workbench.createProject("Distributed Systems Demo");
  await workbench.addDocument();
  await workbench.buildIndex();
  await workbench.retrieveAndJudge();
  if (process.env.CONTEXTBENCH_CAPTURE === "1") {
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({
      path: "../docs/images/query-debugger.png",
      fullPage: false,
    });
  }
  await workbench.runExperiment("browser-hybrid", "HYBRID");
  await workbench.runExperiment("browser-bm25", "BM25");
  await workbench.compareExperiments();
});
