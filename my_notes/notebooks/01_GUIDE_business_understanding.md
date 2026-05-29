# 01 Business Understanding

## Objective
Formalize the problem, define success metrics, identify risks, and align on what "good" looks like.

**You are NOT allowed to look at the data yet.** This phase is about understanding the *business* context, not the dataset.

---

## Part A: Problem Framing

### Questions to Answer

1. **What is the core problem?**
   - We're automating digit recognition for ZIP codes from scanned envelopes.
   - Why is this important? (reduces manual labor, improves speed, reduces errors)

2. **Who are the stakeholders?**
   - Mailroom staff? Logistics operators? Customers?
   - What do they care about? Accuracy? Speed? Ease of use?

3. **What happens if the model is wrong?**
   - If we misclassify a digit, the envelope goes to the wrong destination.
   - Cost of a false positive? Cost of a false negative?
   - Are they equally bad, or is one worse?

4. **What are the constraints?**
   - How fast does it need to run? (real-time, batch, offline?)
   - How much training data do we have or need?
   - What compute resources are available?
   - Does the model need to be interpretable to staff?

5. **What would success look like?**
   - A single number (e.g., 95% accuracy)? Or multiple criteria?
   - Per-digit accuracy? Overall accuracy? Something else?

### Your Task
Write 2–3 paragraphs answering these questions. Be specific.

---

## Part B: Success Criteria and Metrics

### Define Your Metrics

1. **Primary metric:** Which metric matters most?
   - Accuracy? Precision? Recall? F1?
   - If classes are balanced (and MNIST is), why might you still care about per-class metrics?

2. **Acceptable threshold:** What value of the primary metric is "good enough"?
   - 90%? 95%? 99%?
   - Why that number? Is it realistic?

3. **Secondary metrics:** What else matters?
   - Inference speed?
   - Model size?
   - Interpretability?
   - Robustness to unusual handwriting?

4. **Trade-off questions:**
   - If you can improve accuracy by 1% but double training time, is it worth it?
   - If you can improve accuracy by 1% but require 10x more data, is it feasible?

### Your Task
Create a table:
| Metric | Importance | Target | Rationale |
|--------|-----------|--------|-----------|
| [e.g., Overall Accuracy] | [High/Med/Low] | [e.g., ≥95%] | [e.g., baseline business requirement] |
| ... | ... | ... | ... |

---

## Part C: Assumptions and Risks

### Assumptions
List your assumptions about the problem:
- "We assume all digits are roughly the same size."
- "We assume the handwriting is reasonably clear (not heavily degraded)."
- "We assume class balance is not an issue."
- etc.

Which assumptions might be wrong? How would you check them in Phase 2?

### Risks
If these assumptions break, what happens?
- Risk 1: "If handwriting quality varies wildly, the model may fail on poor images."
  - Mitigation: Collect diverse training data; test on degraded images.
- Risk 2: "If one digit is much rarer in real data than in MNIST, the model may not handle it well."
  - Mitigation: Monitor per-digit performance; consider reweighting if needed.

### Your Task
Write down 3–5 key assumptions and 3–5 key risks.

---

## Part D: Non-Goals

What are you **NOT** trying to do?
- NOT trying to achieve 99.5% accuracy (that's beyond learning project scope).
- NOT trying to handle segmentation (extracting individual digits from a full image).
- NOT trying to deploy to production (but design as if we could).
- etc.

### Your Task
List 3–5 explicit non-goals to keep scope manageable.

---

## Part E: Summary

Write a one-paragraph summary of the problem:
- **What:** Classify handwritten digits 0–9.
- **Why:** Automate mail sorting to reduce manual effort.
- **How:** Build a machine learning model trained on MNIST.
- **Success:** Achieve ≥95% accuracy with strong per-digit recall.
- **Constraints:** Model must be fast, interpretable, and robust to handwriting variation.

---

## Output for This Phase

- [ ] Problem statement (2–3 sentences).
- [ ] Success metrics table.
- [ ] 3–5 assumptions documented.
- [ ] 3–5 key risks identified.
- [ ] Non-goals explicitly listed.

**When you're done, move to notebook 02 (Data Understanding).**

---

## Key Habits

1. **Don't peek at the data yet.** Your problem framing should come from business logic, not data exploration.
2. **Be specific.** "Good accuracy" is vague. "≥95% overall accuracy and ≥90% per-digit recall" is concrete.
3. **Justify your metrics.** Why does this metric matter for this business problem?
4. **Document assumptions.** They ground your later decisions.
