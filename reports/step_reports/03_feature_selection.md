# Step 3: Feature selection

## Goal
Keep a small set of informative, non-redundant features for the GAN and the IDS models.

## Method (`src/features.py`)
All fitting uses the **training split** only (stratified subsample: at most 30,000 flows per family,
all 597 Web flows). The validation split is used only to choose the number of features. The test
split is not used.
1. Candidates: the 70 non-constant features minus `Dst Port` (D5).
2. Redundancy filter: for every pair with |Spearman ρ| ≥ 0.95, drop the feature with the lower
   mutual information (MI) with the family label.
3. Ranking: mean of the MI rank (filter method) and the Random Forest importance rank (embedded method).
4. Number of features k: validation macro-F1 of a Random Forest trained on the top-k features; keep
   the smallest k within 0.005 of the best score.

## Justification
- Redundant features add no information, slow training and split importance between "twins",
  which makes explanations misleading.
- A GAN must model the joint distribution of all features: fewer, non-redundant dimensions make
  this feasible with ~600 training flows.
- Two complementary ranking methods avoid depending on the biases of one (MI ignores interactions;
  impurity importance favours high-cardinality features).
- Choosing k on validation, not on train, avoids an optimistic choice; parsimony favours simpler,
  faster, more explainable models.

## Results
_Fill in from the Colab run (sections 3, 5, 7, 8, 9)._
- `Dst Port` rank with all 70 features: ___ (importance ___)
- Redundant features dropped: ___ (69 candidates → ___)
- Chosen k: ___ · validation macro-F1: ___ · Web F1: ___
- Effect of adding `Dst Port`: macro-F1 ___ → ___
- Selected features: ___

### Local test on data slices (indicative only)
`Dst Port` ranked 2nd of 70 (importance 0.088 vs 0.014 for a uniform share). 26 of 69 candidates
were redundant (e.g. `Tot Fwd Pkts` ↔ `Subflow Fwd Pkts`, `Pkt Len Std` ↔ `Pkt Len Var`, ρ = 1).
