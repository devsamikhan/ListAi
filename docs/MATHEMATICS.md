# ListAI — Mathematical Foundations & Derivations

This document provides rigorous mathematical derivations and explanations for all numerical operations implemented in **ListAI**, covering both the **Pure Python List Engine** and the independent **NumPy Engine**.

---

## 1. Network Topology & Notation

We denote an $L$-layer Multilayer Perceptron (MLP) trained on $N$ batch samples with input dimensionality $D_{\text{in}} = 4$ and $C = 3$ output classes:

- **Input Dimension:** $D_{\text{in}} = 4$
- **Hidden Layer 1:** $H_1 = 16$ units, Activation: $\text{ReLU}$
- **Hidden Layer 2:** $H_2 = 8$ units, Activation: $\text{ReLU}$
- **Output Layer:** $C = 3$ units, Activation: $\text{Softmax}$

### Dimensions:
- Batch input: $X \in \mathbb{R}^{N \times 4}$
- Layer 1: $W^{[1]} \in \mathbb{R}^{4 \times 16}$, $b^{[1]} \in \mathbb{R}^{16}$
- Layer 2: $W^{[2]} \in \mathbb{R}^{16 \times 8}$, $b^{[2]} \in \mathbb{R}^{8}$
- Layer 3: $W^{[3]} \in \mathbb{R}^{8 \times 3}$, $b^{[3]} \in \mathbb{R}^{3}$

---

## 2. Forward Propagation

For each layer $l \in \{1, 2, 3\}$, the pre-activation $Z^{[l]}$ is given by the affine transformation:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + \mathbf{1}_N (b^{[l]})^T$$

where $A^{[0]} = X$ and $\mathbf{1}_N$ represents broadcasting the bias row-vector across all $N$ batch samples.

### 2.1 Hidden Layer Activations ($\text{ReLU}$)
For layers 1 and 2:
$$A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]})$$

The element-wise derivative is:
$$\text{ReLU}'(z) = \begin{cases} 1.0 & \text{if } z > 0 \\ 0.0 & \text{if } z \le 0 \end{cases}$$

### 2.2 Output Layer Activation ($\text{Softmax}$)
For sample $i$ and class $c \in \{1, \dots, C\}$:
$$\hat{y}_{i,c} = A^{[3]}_{i,c} = \frac{\exp(Z^{[3]}_{i,c} - M_i)}{\sum_{k=1}^C \exp(Z^{[3]}_{i,k} - M_i)}$$

where $M_i = \max_{k} Z^{[3]}_{i,k}$ is the row maximum. 

**Numerical Stability Proof:**
$$\frac{\exp(z_{i,c} - M_i)}{\sum_{k} \exp(z_{i,k} - M_i)} = \frac{\exp(z_{i,c}) \exp(-M_i)}{\sum_{k} \exp(z_{i,k}) \exp(-M_i)} = \frac{\exp(z_{i,c})}{\sum_{k} \exp(z_{i,k})}$$
Because $z_{i,c} - M_i \le 0$, the argument to $\exp$ is guaranteed to be non-positive, completely preventing floating-point overflow (`OverflowError: math range error` in Python).

---

## 3. Loss Function: Categorical Cross-Entropy

For $N$ samples and $C$ one-hot encoded classes:
$$L(Y, \hat{Y}) = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{i,c} \ln(\hat{Y}_{i,c})$$

To guard against $\ln(0) = -\infty$, predictions are clamped to $[\varepsilon, 1 - \varepsilon]$ with $\varepsilon = 10^{-15}$.

---

## 4. Analytical Backpropagation Derivations

### 4.1 Combined Softmax and Cross-Entropy Gradient
Let $z = Z^{[3]}_{i, :}$ and $a = A^{[3]}_{i, :}$. The loss for a single sample is:
$$L_i = - \sum_{k=1}^C y_k \ln(a_k)$$

The Jacobian of the Softmax function is:
$$\frac{\partial a_k}{\partial z_c} = \begin{cases} a_k (1 - a_k) & \text{if } k = c \\ - a_k a_c & \text{if } k \neq c \end{cases} = a_k (\delta_{kc} - a_c)$$

Applying the multivariate chain rule:
$$\frac{\partial L_i}{\partial z_c} = \sum_{k=1}^C \frac{\partial L_i}{\partial a_k} \frac{\partial a_k}{\partial z_c} = - \sum_{k=1}^C \frac{y_k}{a_k} \left[ a_k (\delta_{kc} - a_c) \right]$$
$$= - \sum_{k=1}^C y_k (\delta_{kc} - a_c) = - y_c + a_c \sum_{k=1}^C y_k$$

Since $Y_i$ is a valid one-hot target distribution, $\sum_{k=1}^C y_k = 1$. Therefore:
$$\frac{\partial L_i}{\partial z_c} = a_c - y_c$$

For the entire batch matrix:
$$\frac{\partial L}{\partial Z^{[3]}} = \frac{1}{N} (A^{[3]} - Y) \in \mathbb{R}^{N \times C}$$

### 4.2 Parameter Gradients for Layer 3
$$\frac{\partial L}{\partial W^{[3]}} = (A^{[2]})^T \frac{\partial L}{\partial Z^{[3]}} \in \mathbb{R}^{8 \times 3}$$
$$\frac{\partial L}{\partial b^{[3]}} = \sum_{i=1}^N \left(\frac{\partial L}{\partial Z^{[3]}}\right)_{i, :} \in \mathbb{R}^{3}$$

### 4.3 Backpropagation to Hidden Layer 2
By the chain rule:
$$\frac{\partial L}{\partial A^{[2]}} = \frac{\partial L}{\partial Z^{[3]}} (W^{[3]})^T \in \mathbb{R}^{N \times 8}$$
$$\frac{\partial L}{\partial Z^{[2]}} = \frac{\partial L}{\partial A^{[2]}} \odot \text{ReLU}'(Z^{[2]}) \in \mathbb{R}^{N \times 8}$$
$$\frac{\partial L}{\partial W^{[2]}} = (A^{[1]})^T \frac{\partial L}{\partial Z^{[2]}} \in \mathbb{R}^{16 \times 8}$$
$$\frac{\partial L}{\partial b^{[2]}} = \sum_{i=1}^N \left(\frac{\partial L}{\partial Z^{[2]}}\right)_{i, :} \in \mathbb{R}^{8}$$

### 4.4 Backpropagation to Hidden Layer 1
$$\frac{\partial L}{\partial A^{[1]}} = \frac{\partial L}{\partial Z^{[2]}} (W^{[2]})^T \in \mathbb{R}^{N \times 16}$$
$$\frac{\partial L}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \odot \text{ReLU}'(Z^{[1]}) \in \mathbb{R}^{N \times 16}$$
$$\frac{\partial L}{\partial W^{[1]}} = (X)^T \frac{\partial L}{\partial Z^{[1]}} \in \mathbb{R}^{4 \times 16}$$
$$\frac{\partial L}{\partial b^{[1]}} = \sum_{i=1}^N \left(\frac{\partial L}{\partial Z^{[1]}}\right)_{i, :} \in \mathbb{R}^{16}$$

---

## 5. Numerical Gradient Verification (Finite Differences)

To mathematically certify backpropagation without trusting automatic differentiation, two-sided finite difference approximations are computed:

$$g_{\text{num}}(\theta) = \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}, \quad \varepsilon = 10^{-5}$$

The relative discrepancy norm is evaluated:
$$\text{Relative Error} = \frac{\|g_{\text{ana}} - g_{\text{num}}\|_2}{\max(\|g_{\text{ana}}\|_2 + \|g_{\text{num}}\|_2, 10^{-12})}$$

A relative error $< 10^{-6}$ confirms exact analytical backpropagation.
