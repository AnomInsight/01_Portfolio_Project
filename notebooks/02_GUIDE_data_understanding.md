# 02 Data Understanding

## Objective
Explore the MNIST dataset systematically. Learn its structure, quirks, patterns, and challenges.

**Now you can look at the data for the first time.**

---

## Part A: Load and Inspect

### Task 1: Load the MNIST Dataset
Questions to answer:
1. Where does MNIST come from? (keras.datasets, sklearn, or direct download?)
2. How many total samples are there?
3. What is the shape of each image? (Should be 28×28 pixels)
4. What is the pixel value range? (0–255 or 0–1?)
5. Are there 10 classes (digits 0–9)? Confirm.

### Your Checklist
- [ ] Loaded dataset successfully.
- [ ] Know exact shape and data type.
- [ ] Know pixel value range.
- [ ] Know number of classes and samples per class.

---

## Part B: Class Balance

### Task 2: Examine Class Distribution
Questions to answer:
1. How many samples per digit?
2. Is the dataset balanced? (Should be roughly equal, but check!)
3. Are any classes overrepresented or underrepresented?
4. If imbalanced, how would you handle it in preprocessing?

### Your Checklist
- [ ] Counted samples per class.
- [ ] Created a bar chart showing class distribution.
- [ ] Confirmed or noted any imbalance.

---

## Part C: Visual Exploration

### Task 3: Visualize Sample Images
Questions to answer:
1. Show 2–3 examples from each digit class (0–9). Do they look like what you expect?
2. Do any digits look ambiguous or hard to read?
3. Are there obvious patterns or variations in handwriting style?
4. Which pairs of digits look most similar? (e.g., 4 vs 9, 3 vs 5, 1 vs 7)

### Your Checklist
- [ ] Displayed a 10×5 grid of example digits (or similar).
- [ ] Noted which digits look similar or confusing.
- [ ] Identified at least 3 digit pairs that might be commonly confused.

---

## Part D: Statistical Summary

### Task 4: Pixel Statistics
Questions to answer:
1. What is the mean pixel intensity across all images?
2. What is the std dev of pixel intensity?
3. What is the min and max pixel value?
4. How sparse are the images? (What % of pixels are exactly 0?)
5. Does pixel intensity distribution differ by digit?

### Your Checklist
- [ ] Computed mean, std, min, max for the dataset.
- [ ] Checked sparsity.
- [ ] Plotted pixel intensity distributions (histogram).

---

## Part E: Data Quality Checks

### Task 5: Look for Issues
Questions to answer:
1. Are there any completely blank images?
2. Are there any perfect duplicates?
3. Are there any obviously mislabeled samples?
4. What's the range of "darkness" across handwriting? (Some digits bolder than others?)

### Your Checklist
- [ ] Manually inspected 20–30 random samples for obvious errors.
- [ ] Checked for duplicates using a simple hash or comparison method.
- [ ] Noted any concerning patterns.

---

## Part F: Hypothesis Generation

### Task 6: Form Hypotheses for Modeling
Based on what you've observed, write down predictions:

1. **Easy predictions:** Which digits do you think will be easiest to classify and why?
   - Example: "0 and 1 will be easy because they have distinct shapes."

2. **Hard predictions:** Which will be hardest and why?
   - Example: "4 and 9 will be confused because they share similar curved strokes."

3. **Feature predictions:** What patterns do you think the model should learn?
   - Example: "The model should learn that 1, 7, I are vertical; 0, 8, 9 are rounded."

4. **Dataset predictions:** Are there any quirks or biases in MNIST that a real-world system wouldn't have?
   - Example: "All digits are centered and same-sized; real handwriting varies."

### Your Checklist
- [ ] Written down 3–5 hypotheses about what's easy/hard.
- [ ] Justified each hypothesis based on your visual exploration.

---

## Part G: Summary Statistics

Create a summary table:
| Attribute | Value |
|-----------|-------|
| Total samples | ? |
| Classes | ? |
| Image shape | ? |
| Pixel range | ? |
| Mean pixel intensity | ? |
| Class balance | Balanced / Imbalanced / [describe] |
| Missing values | ? |
| Duplicates | ? |

---

## Output for This Phase

- [ ] Dataset loaded and basic stats printed.
- [ ] Class distribution visualized.
- [ ] 10 (or 20–30) example images displayed and labeled.
- [ ] Statistical summary created (mean, std, min, max, sparsity).
- [ ] Data quality checks completed (duplicates, blanks, outliers).
- [ ] 3–5 hypotheses written about hard/easy digits.
- [ ] Summary table filled in.

**When you're done, update CRISP_DM_LOG.md with your findings, then move to notebook 03.**

---

## Key Habits

1. **Visualize before you model.** You learn more from a chart than from a summary stat.
2. **Don't assume MNIST is clean.** Even famous datasets have quirks.
3. **Form hypotheses before modeling.** Later, you can check if the model agrees with you.
4. **Document surprising findings.** These become talking points in your portfolio.
