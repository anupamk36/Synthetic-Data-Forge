import { test, expect } from "@playwright/test";
import { SAMPLE_FHIR_TYPES, SAMPLE_FHIR_RESPONSE } from "./fixtures/sample-data";

test.describe("FHIR Generation Wizard", () => {
  test.beforeEach(async ({ page }) => {
    // Mock the FHIR resource types endpoint
    await page.route("**/api/v1/medical/fhir/resource-types", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(SAMPLE_FHIR_TYPES),
      });
    });

    // Mock the FHIR generate endpoint
    await page.route("**/api/v1/medical/fhir/generate", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(SAMPLE_FHIR_RESPONSE),
      });
    });

    await page.goto("/medical/fhir");
  });

  test("wizard steps are visible", async ({ page }) => {
    // Verify the wizard UI is present with step indicators
    const stepIndicators = page.locator("[data-step], [class*='step'], [role='tablist']");
    await expect(stepIndicators.first()).toBeVisible();
  });

  test("resource types are loaded and selectable", async ({ page }) => {
    // Verify resource types from the mocked API are displayed
    const patientOption = page.getByText("Patient");
    await expect(patientOption).toBeVisible();

    const encounterOption = page.getByText("Encounter");
    await expect(encounterOption).toBeVisible();
  });

  test("full generation workflow completes", async ({ page }) => {
    // Step 1: Select resource types
    const patientOption = page.getByText("Patient").first();
    await patientOption.click();

    const encounterOption = page.getByText("Encounter").first();
    await encounterOption.click();

    // Look for a next/continue button to advance
    const nextButton = page.getByRole("button", { name: /next|continue/i });
    if (await nextButton.isVisible()) {
      await nextButton.click();
    }

    // Step 2: Configure generation options (if a configure step exists)
    const configureStep = page.getByText(/configure|options|settings/i).first();
    if (await configureStep.isVisible().catch(() => false)) {
      // Fill in count if input exists
      const countInput = page.getByLabel(/count|number|records/i);
      if (await countInput.isVisible().catch(() => false)) {
        await countInput.fill("100");
      }
    }

    // Step 3: Click generate
    const generateButton = page.getByRole("button", {
      name: /generate/i,
    });
    await generateButton.click();

    // Verify results section appears with stats from mocked response
    const resultsSection = page.getByText(/complete|results|generated/i).first();
    await expect(resultsSection).toBeVisible();
  });

  test("displays generation statistics after completion", async ({ page }) => {
    // Select a resource type and generate
    const patientOption = page.getByText("Patient").first();
    await patientOption.click();

    const generateButton = page.getByRole("button", {
      name: /generate/i,
    });
    await generateButton.click();

    // Verify statistics from the mocked response are displayed
    const totalStat = page.getByText("100");
    await expect(totalStat).toBeVisible();
  });
});
