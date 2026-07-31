import { test, expect } from "@playwright/test";

test.describe("Medical-First Landing Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("page loads without redirect", async ({ page }) => {
    // Should stay on root URL, not redirect to /generate/single
    await expect(page).toHaveURL("/");
  });

  test('hero text "Generate Compliant Clinical" is visible', async ({
    page,
  }) => {
    const heroText = page.getByText("Generate Compliant Clinical");
    await expect(heroText).toBeVisible();
  });

  test("three clinical feature cards exist", async ({ page }) => {
    const fhirCard = page.getByText("FHIR Generator");
    const trialsCard = page.getByText("Clinical Trials");
    const imagingCard = page.getByText("Medical Imaging");

    await expect(fhirCard).toBeVisible();
    await expect(trialsCard).toBeVisible();
    await expect(imagingCard).toBeVisible();
  });

  test("feature cards link to correct URLs", async ({ page }) => {
    const fhirLink = page.getByRole("link", { name: /FHIR Generator/i });
    const trialsLink = page.getByRole("link", { name: /Clinical Trials/i });
    const imagingLink = page.getByRole("link", { name: /Medical Imaging/i });

    await expect(fhirLink).toHaveAttribute("href", /\/medical\/fhir/);
    await expect(trialsLink).toHaveAttribute("href", /\/medical\/trials/);
    await expect(imagingLink).toHaveAttribute("href", /\/medical\/imaging/);
  });

  test('secondary section "Also Generates Any Tabular Data" exists', async ({
    page,
  }) => {
    const sectionHeading = page.getByText("Also Generates Any Tabular Data");
    await expect(sectionHeading).toBeVisible();
  });

  test("generic feature cards exist", async ({ page }) => {
    const singleTableCard = page.getByText("Single Table");
    const multiTableCard = page.getByText("Multi-Table");

    await expect(singleTableCard).toBeVisible();
    await expect(multiTableCard).toBeVisible();
  });
});
