import { test, expect } from "@playwright/test";
import {
  SAMPLE_UPLOAD_RESPONSE,
  SAMPLE_PRIVACY_REPORT,
} from "./fixtures/sample-data";

test.describe("Privacy Audit Workflow", () => {
  test.beforeEach(async ({ page }) => {
    // Mock the upload endpoint
    await page.route("**/api/v1/upload", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(SAMPLE_UPLOAD_RESPONSE),
      });
    });

    // Mock the privacy report endpoint
    await page.route("**/api/v1/privacy/report", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(SAMPLE_PRIVACY_REPORT),
      });
    });

    await page.goto("/analyze/privacy");
  });

  test("upload sections exist on the page", async ({ page }) => {
    // There should be upload areas for original and synthetic data
    const uploadAreas = page.getByText(/upload|drop|select.*file/i);
    await expect(uploadAreas.first()).toBeVisible();
  });

  test("shows original data upload section", async ({ page }) => {
    const originalSection = page.getByText(/original|real|source/i).first();
    await expect(originalSection).toBeVisible();
  });

  test("shows synthetic data upload section", async ({ page }) => {
    const syntheticSection = page.getByText(/synthetic|generated/i).first();
    await expect(syntheticSection).toBeVisible();
  });

  test("metric cards appear after audit", async ({ page }) => {
    // Trigger the audit by clicking the audit/analyze button
    const auditButton = page.getByRole("button", {
      name: /audit|analyze|run/i,
    });
    await auditButton.click();

    // Verify DCR metric card
    const dcrMetric = page.getByText(/DCR|Distance to Closest Record/i);
    await expect(dcrMetric).toBeVisible();

    // Verify k-anonymity metric card
    const kAnonMetric = page.getByText(/k-anonymity/i);
    await expect(kAnonMetric).toBeVisible();

    // Verify epsilon metric card
    const epsilonMetric = page.getByText(/epsilon/i);
    await expect(epsilonMetric).toBeVisible();
  });

  test("compliance badge appears after audit", async ({ page }) => {
    // Trigger the audit
    const auditButton = page.getByRole("button", {
      name: /audit|analyze|run/i,
    });
    await auditButton.click();

    // Verify compliance badge shows compliant status
    const complianceBadge = page.getByText(/compliant|low.?risk|passed/i);
    await expect(complianceBadge).toBeVisible();
  });

  test("download report button exists after audit", async ({ page }) => {
    // Trigger the audit
    const auditButton = page.getByRole("button", {
      name: /audit|analyze|run/i,
    });
    await auditButton.click();

    // Verify download button is present
    const downloadButton = page.getByRole("button", {
      name: /download|export|save.*report/i,
    });
    await expect(downloadButton).toBeVisible();
  });
});
