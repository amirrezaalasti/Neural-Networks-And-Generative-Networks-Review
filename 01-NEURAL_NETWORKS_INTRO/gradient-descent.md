# **Gradient Descent & Backward Propagation**

## **1. Gradient Descent Algorithm**
Gradient Descent is an optimization algorithm used to minimize the **loss function** by iteratively updating model parameters in the **opposite direction of the gradient**.

### **Mathematical Formulation**
Given a loss function $ J(\theta) $, the update rule for parameters $ \theta $ is:

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

where:
- $ \alpha $ is the **learning rate** (step size),
- $ \nabla J(\theta) $ is the **gradient** of the loss function.

---

## **2. Types of Gradient Descent**

### **a) Stochastic Gradient Descent (SGD)**
Instead of computing the gradient using the entire dataset, **SGD** updates parameters after processing each **single** training sample.

**Update Rule:**
$$
\theta := \theta - \alpha \nabla J(\theta; x_i, y_i)
$$
where $ (x_i, y_i) $ is a single training example.

**Intuition:**
- Faster updates.
- High variance in updates → causes **noisy convergence**.
- Can escape **local minima** but may oscillate.

---

### **b) Mini-batch Gradient Descent**
Instead of using one example (**SGD**) or the entire dataset (**Batch GD**), Mini-batch GD updates parameters using **small subsets** of data.

**Update Rule:**
$$
\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla J(\theta; x_i, y_i)
$$
where $ m $ is the batch size.

**Intuition:**
- Balances stability and speed.
- Reduces variance while keeping computational efficiency.

---

### **c) Momentum**
Momentum helps **accelerate** gradient descent by maintaining a moving average of past gradients, reducing oscillations.

**Update Rule:**
$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta)
$$
$$
\theta := \theta - \alpha v_t
$$
where:
- $ v_t $ is the **velocity term** (gradient memory),
- $ \beta $ is the **momentum coefficient** (e.g., 0.9).

**Intuition:**
- Helps escape **local minima** faster.
- Reduces oscillations in steep directions.

---

### **d) Nesterov Accelerated Gradient (NAG)**
An improvement over **Momentum**, NAG **looks ahead** before computing the gradient.

**Update Rule:**
$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta - \beta v_{t-1})
$$
$$
\theta := \theta - \alpha v_t
$$

**Intuition:**
- Reduces excessive updates by anticipating the next step.
- Improves convergence speed.

---

### **e) AdaGrad (Adaptive Gradient)**
AdaGrad adapts the learning rate for each parameter **individually**, decreasing it for frequently updated parameters.

**Update Rule:**
$$
G_t = G_{t-1} + \nabla J(\theta)^2
$$
$$
\theta := \theta - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla J(\theta)
$$

**Intuition:**
- Good for **sparse data** (e.g., NLP).
- Learning rate decays too aggressively.

---

### **f) Adam (Adaptive Moment Estimation)**
Combines **Momentum** and **RMSProp**, maintaining both **exponentially weighted averages** of past gradients and squared gradients.

**Update Rule:**
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla J(\theta)
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla J(\theta))^2
$$
$$
\theta := \theta - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t
$$

where:
- $ m_t $ is the **momentum** term,
- $ v_t $ is the **adaptive learning rate** term.

**Intuition:**
- Works well across **different datasets**.
- Adaptively adjusts learning rate.

---

## **3. Backward Propagation**
Backward Propagation (**Backprop**) computes gradients using the **chain rule** to update weights.

### **Example Computation**
Consider a simple function:
$$
z = x^2 + y^2
$$
Given $ x = 3, y = 4 $, compute $ \frac{\partial z}{\partial x} $ and $ \frac{\partial z}{\partial y} $.

1. **Partial derivatives:**
   $$
   \frac{\partial z}{\partial x} = 2x = 6, \quad \frac{\partial z}{\partial y} = 2y = 8
   $$

2. **Gradient vector:**
   $$
   \nabla z = (6, 8)
   $$

3. **Weight update (SGD, learning rate = 0.1):**
   $$
   x := x - 0.1 \times 6 = 2.4
   $$
   $$
   y := y - 0.1 \times 8 = 3.2
   $$

### **MATLAB Code for Backpropagation Example**
```matlab
% Define function
x = 3;
y = 4;
alpha = 0.1; % Learning rate

% Compute gradients
grad_x = 2 * x;
grad_y = 2 * y;

% Update weights using SGD
x_new = x - alpha * grad_x;
y_new = y - alpha * grad_y;

% Display results
fprintf('Updated x: %.2f\\n', x_new);
fprintf('Updated y: %.2f\\n', y_new);
