# Taylor Approximation in Neural Network Pruning

## Experimental Setup

- **Datasets**:  
  - MNIST  
  - Fashion-MNIST

- **Models**:  
  - LeNet-5 (for MNIST)  
  - Custom CNN (for Fashion-MNIST)

- **Baseline**:  
  - Accuracy of the full (unpruned) model

- **Evaluation Metrics**:
  - **Accuracy**: Classification performance after pruning
  - **Parameters**: Total number of model weights
  - **FLOPs**: Floating Point Operations, representing computational cost

---

## Experiments

### 1. LeNet-5 + Taylor Pruning (MNIST)
- Pruning levels: Full model vs 50% Pruned vs 70% Pruned
- Objective: Evaluate the trade-off between compression and performance using first-order Taylor approximation.

### 2. Custom CNN + Taylor Pruning (Fashion-MNIST)
- Pruning levels: Full model vs 50% Pruned vs 70% Pruned
- Objective: Assess generalization of Taylor pruning on a more complex dataset and slightly deeper CNN.

