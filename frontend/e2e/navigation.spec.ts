import { test, expect } from "@playwright/test";

test.describe("Sidebar Navigation", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test('sidebar renders with "Clinical Data Forge" branding', async ({
    page,
  }) => {
    const branding = page.getByText("Clinical Data Forge");
    await expect(branding).toBeVisible();
  });

  test('"Clinical Data" section is first in sidebar navigation', async ({
    page,
  }) => {
    const sidebar = page.getByRole("navigation");
    // The first navigation section should be Clinical Data
    const sections = sidebar.locator("[data-section], h3, h4, [class*='section']");
    const firstSection = sections.first();
    await expect(firstSection).toContainText(/Clinical/i);
  });

  test("FHIR Generator link navigates correctly", async ({ page }) => {
    const fhirLink = page.getByRole("link", { name: /FHIR/i }).first();
    await fhirLink.click();
    await expect(page).toHaveURL(/\/medical\/fhir/);
  });

  test("Clinical Trials link navigates correctly", async ({ page }) => {
    const trialsLink = page.getByRole("link", { name: /Trials/i }).first();
    await trialsLink.click();
    await expect(page).toHaveURL(/\/medical\/trials/);
  });

  test("Single Table link navigates correctly", async ({ page }) => {
    const singleLink = page
      .getByRole("link", { name: /Single Table/i })
      .first();
    await singleLink.click();
    await expect(page).toHaveURL(/\/generate\/single/);
  });

  test("Privacy Audit link navigates correctly", async ({ page }) => {
    const privacyLink = page
      .getByRole("link", { name: /Privacy/i })
      .first();
    await privacyLink.click();
    await expect(page).toHaveURL(/\/analyze\/privacy/);
  });

  test("active state highlights when on a page", async ({ page }) => {
    // Navigate to FHIR page
    const fhirLink = page.getByRole("link", { name: /FHIR/i }).first();
    await fhirLink.click();
    await expect(page).toHaveURL(/\/medical\/fhir/);

    // The active link should have an active/highlighted class or aria-current
    const activeLink = page.getByRole("link", { name: /FHIR/i }).first();
    const classAttr = await activeLink.getAttribute("class");
    const ariaCurrent = await activeLink.getAttribute("aria-current");

    // Either the link has an active-indicating class or aria-current="page"
    const isActive =
      ariaCurrent === "page" ||
      (classAttr &&
        (classAttr.includes("active") || classAttr.includes("bg-")));
    expect(isActive).toBeTruthy();
  });

  test("command palette opens with Cmd+K and shows clinical commands first", async ({
    page,
  }) => {
    // Open command palette with Cmd+K (Meta+K)
    await page.keyboard.press("Meta+k");

    // Command palette should be visible
    const palette = page.getByRole("dialog");
    await expect(palette).toBeVisible();

    // First group or items should be clinical-related
    const firstItem = palette.locator("[cmdk-item], [role='option']").first();
    await expect(firstItem).toContainText(/Clinical|FHIR|Medical/i);
  });
});
