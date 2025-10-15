# Responsible AI (RAI) Report for Flight Price Prediction Model

This document outlines the steps taken to ensure the Indigo Flight Price Prediction model was developed and evaluated responsibly, focusing on fairness, transparency, and accountability.

---

## 1. Transparency and Explainability

**Objective:** To ensure that the model's decision-making process is understandable to stakeholders.

* **Global Explainability (SHAP):** We used the SHAP (SHapley Additive exPlanations) library to understand the model's behavior across all predictions. The SHAP summary plot identified `Class` (Economy vs. Premium Economy), `Year`, and `Aircraft_Type` as the most influential factors in price prediction. This confirms that the model has learned logical, business-relevant patterns.

* **Local Explainability (LIME):** We used LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions. This allows us to inspect why a specific flight was given a certain price, breaking down the prediction into the contributions of its features. This builds trust and allows for debugging of anomalous predictions.

---

## 2. Fairness and Bias

**Objective:** To identify and quantify any systematic biases in the model's performance across different subgroups.

* **Sensitive Attribute:** For our business context, we identified **`Class`** (Economy vs. Premium Economy) as a sensitive attribute to audit for performance bias. The goal was to ensure the model's prediction accuracy was consistent for all customer groups.

* **Fairness Audit:** Using the `Fairlearn` library, we conducted a fairness audit. The key findings were:
    * **RMSE for Economy Class:** ~₹2016
    * **RMSE for Premium Economy Class:** ~₹1838
    * **Performance Disparity:** The model's average prediction error is **~₹178 higher** for Economy class flights.

* **Conclusion:** The audit revealed a performance bias. The model is less reliable (i.e., less "fair" in terms of performance) for customers booking standard Economy tickets.

* **Proposed Mitigation:**
    1.  **Pre-processing:** Apply reweighting to the training data to give more importance to the underperforming 'Economy' group.
    2.  **In-processing:** Use a fairness-aware training algorithm that constrains the model to minimize error disparity across classes.
    3.  **Post-processing:** Apply a calibrated adjustment to the final predictions for Economy class flights to correct for the systematic error.

---

## 3. Privacy and Data Governance

* **Data Used:** The model was trained exclusively on anonymized flight data, including routes, schedules, aircraft types, and booking channels.
* **No Personal Data:** The dataset does **not** contain any Personally Identifiable Information (PII) such as passenger names, contact details, or payment information.
* **User Input:** The interactive dashboard only requires general flight characteristics for prediction and does not ask for or store any user-specific data.

---

## 4. Accountability and Human Oversight

* **Intended Use:** This model is intended as a tool to provide **price estimates** and understand market dynamics. It is not designed to be a fully autonomous pricing system without human oversight.
* **Model Card:** All key information, including performance metrics (RMSE), fairness audit results, and influential features, is documented.
* **CI/CD Pipeline:** An automated CI/CD pipeline is in place to test the model's integrity and build process whenever code changes are made, ensuring reliability and accountability.
