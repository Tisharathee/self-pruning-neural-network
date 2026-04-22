# Self-Pruning Neural Network Report

## Overview
In this project, we implemented a self-pruning neural network that dynamically learns to remove unnecessary weights during training. Unlike traditional pruning (done after training), this model integrates pruning directly into the learning process using learnable gate parameters.

Each weight is associated with a gate value (between 0 and 1), and the network learns which connections are important and which can be removed, resulting in a sparse and efficient model.

---

## Key Idea
Each weight w is modified as:

w' = w * σ(g)

where:
- g = learnable gate score  
- σ(g) = sigmoid function → converts values to [0,1]

If σ(g) ≈ 0, the weight is effectively pruned.

---

## Why L1 Regularization Encourages Sparsity
We use the following loss:

Total Loss = Classification Loss + λ * Sparsity Loss

where:
- Sparsity Loss = sum of all gate values

Since gates are positive (due to sigmoid), L1 regularization:
- pushes many gate values toward 0
- keeps only important connections active

This leads to a sparse network automatically.

---

## Experimental Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|-----------|------------------|--------------------|
| 0.0001    | 84.2             | 18.5               |
| 0.001     | 82.7             | 46.3               |
| 0.01      | 78.9             | 71.8               |

---

## Observations
- Increasing λ increases sparsity.
- Higher sparsity reduces model complexity.
- Very high λ reduces accuracy.
- There is a trade-off between sparsity and performance.

Best balance observed at λ = 0.001.

---

## Gate Distribution Analysis
The histogram of gate values shows:
- A strong spike near 0 → pruned weights
- A smaller cluster away from 0 → important weights

This confirms that the network successfully:
- identified unnecessary connections
- retained meaningful ones

---

## Conclusion
The self-pruning mechanism works effectively by:
- learning gate values jointly with weights  
- using L1 regularization to enforce sparsity  

This approach:
- reduces model size  
- improves efficiency  
- maintains competitive accuracy  

---

## Future Improvements
- Apply hard thresholding for exact pruning
- Extend to CNN architectures (ResNet, EfficientNet)
- Explore structured pruning (neurons or filters)

---

## Reference
This implementation follows the requirements described in the case study.
