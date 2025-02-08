### **Contrastive Learning and Pretext Tasks**
Contrastive learning is a self-supervised learning approach that trains a model by pulling similar data points (positive pairs) together and pushing dissimilar data points (negative pairs) apart in an embedding space.

A **pretext task** in contrastive learning is an artificial learning objective used to train a model without explicit labels. The idea is to create a task that encourages the model to learn meaningful representations from data. Some common pretext tasks include:
- **SimCLR-style augmentations**: Applying transformations (like cropping, flipping, or adding noise) to an image and ensuring the model learns that these transformed versions are similar.
- **Instance discrimination**: Treating each instance as its own class and pushing away all other instances.

---

### **Contrastive Loss**
Contrastive loss is used to train the model by minimizing the distance between similar samples while maximizing the distance between dissimilar samples.

The general **contrastive loss** function for a pair of samples \((x_i, x_j)\) is:

$$
L = (1 - y) \cdot D^2 + y \cdot \max(0, m - D)^2
$$

where:
- \( D \) is the distance between the embeddings of \( x_i \) and \( x_j \).
- \( y = 1 \) if the pair is dissimilar and \( y = 0 \) if the pair is similar.
- \( m \) is a margin that ensures dissimilar pairs are pushed apart.

---

### **Triplet Loss**
Triplet loss is another contrastive loss used in metric learning. Instead of comparing just two samples, it considers three:
- An **anchor** \( A \)
- A **positive sample** \( P \) (similar to the anchor)
- A **negative sample** \( N \) (dissimilar to the anchor)

The goal is to ensure that the positive is closer to the anchor than the negative by a margin \( m \). The **triplet loss** is:

$$
L = \max(0, D(A, P) - D(A, N) + m)
$$

where:
- \( D(A, P) \) is the distance between the anchor and positive.
- \( D(A, N) \) is the distance between the anchor and negative.
- \( m \) is the margin to prevent trivial solutions.

The loss forces the model to ensure:

$$
D(A, P) + m < D(A, N)
$$

meaning the positive sample is always closer than the negative.

---

### **N-Pair Loss**
N-pair loss extends the triplet loss by comparing one anchor to multiple negative samples, improving learning efficiency.

For a batch of samples:
1. Choose an **anchor** \( A \).
2. Select a **positive sample** \( P \).
3. Compare it to **N negatives** \( N_1, N_2, ..., N_N \).

The loss function is based on a softmax function over similarity scores:

$$
L = \log \left( 1 + \sum_{i=1}^{N} e^{D(A, P) - D(A, N_i)} \right)
$$

This encourages the positive pair to be closer while pushing away all negatives.

---

### **Conclusion**
- **Contrastive loss** works with pairs and directly minimizes/maximizes distances.
- **Triplet loss** uses an anchor, positive, and negative to enforce relative distances.
- **N-pair loss** generalizes triplet loss to multiple negatives for better learning efficiency.

These losses are crucial in self-supervised learning, representation learning, and applications like face recognition, retrieval, and clustering.