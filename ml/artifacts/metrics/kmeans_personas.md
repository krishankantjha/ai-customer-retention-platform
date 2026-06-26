# Customer Segmentation Personas & Profiles

This document details the behavioral segments identified using K-Means++ clustering on continuous feature coordinates.

---

## Behavioral Personas Overview

### Cluster 0: Moderate-Value, Budget-Conscious Users
* **Segment Size:** 1,214 customers (21.55% of the training set)
* **Average Churn Rate:** **7.25%**
* **Scaled Behavioral Scores:**
  * Tenure: `-0.092` (Low-to-moderate tenure)
  * Monthly Charges: `-1.454` (Low monthly billing)
  * Ecosystem Services: `-1.315` (Low ecosystem lock-in)
* **Profile Description:** Medium-tenure customers paying low-to-moderate monthly charges with few ecosystem services. This represents a budget-conscious core user base.
* **Retention Strategy:** Trigger Auto-Pay setups and cross-sell technical security add-ons to improve customer lock-in and account stickiness.

---

### Cluster 1: High-Value Premium Cohort
* **Segment Size:** 2,523 customers (44.78% of the training set)
* **Average Churn Rate:** **44.95%**
* **Scaled Behavioral Scores:**
  * Tenure: `-0.713` (Low tenure relative to billing weight)
  * Monthly Charges: `+0.141` (High monthly billing)
  * Ecosystem Services: `-0.057` (Moderate ecosystem lock-in)
* **Profile Description:** Long-tenure customers with high ecosystem service counts and high monthly billing rates. This is the platform's most valuable cohort.
* **Retention Strategy:** Ensure high-priority VIP customer support. Run network hardware checks and offer proactive loyalty credits to mitigate high tariff risk.

---

### Cluster 2: New Churn-Risk Users
* **Segment Size:** 1,897 customers (33.67% of the training set)
* **Average Churn Rate:** **14.39%**
* **Scaled Behavioral Scores:**
  * Tenure: `+1.007` (High tenure segment variance)
  * Monthly Charges: `+0.743` (High monthly billing)
  * Ecosystem Services: `+0.917` (High ecosystem lock-in)
* **Profile Description:** Short-tenure customers with high monthly charges, short-term contract structures, and low ecosystem subscription counts. This represents the highest immediate churn risk.
* **Retention Strategy:** Prioritize direct welcome check-ins, rate audits, and transition campaigns targeting long-term contract conversions.
