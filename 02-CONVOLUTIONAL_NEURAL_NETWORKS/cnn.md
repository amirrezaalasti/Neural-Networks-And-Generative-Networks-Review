## **MLP vs CNN**
| **Problem**                      | **MLP** (Multi-Layer Perceptron) | **CNN** (Convolutional Neural Network) |
|-----------------------------------|--------------------------------|--------------------------------|
| **High number of parameters**     | Each neuron connects to all others → Too many weights | Uses weight-sharing in convolutional layers |
| **Loss of spatial information**   | Flattens images → Loses structure | Preserves spatial relationships |
| **Inefficient learning**          | Independent learning per neuron | Shared filters across locations |
| **Lack of translation invariance**| Position changes → Poor recognition | Translation-invariant feature detection |
| **High computational cost**       | Fully connected layers are expensive | Convolutions reduce computation |

# Convolutional Neural Networks (CNNs)

## Features of CNNs:
1. **Convolutional Layers**: These layers apply convolution operations using filters (kernels) that detect local patterns such as edges, textures, or objects in images.
2. **Pooling Layers**: Pooling (usually max pooling) reduces the spatial size (width and height) of the input, helping to reduce computational cost and control overfitting.
3. **Fully Connected Layers**: After convolution and pooling, fully connected layers perform the classification based on the extracted features.
4. **ReLU Activation**: CNNs often use ReLU activation functions, which introduce non-linearity, enabling the network to learn complex patterns.
5. **Weight Sharing**: Filters (kernels) are shared across the input image, reducing the number of parameters.
6. **Local Receptive Fields**: CNNs look at small local regions of the input, which makes them good at detecting local patterns that can be generalized across the image.

## Output Size Calculation:
Given an input image of size $ W \times H $, filter size $ F \times F $, stride $ S $, and padding $ P $, the output size $ O_W \times O_H $ for width and height can be calculated as:

$$ O_W = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1 $$

$$ O_H = \left\lfloor \frac{H - F + 2P}{S} \right\rfloor + 1 $$

Where:
- $ W $ is the input width
- $ H $ is the input height
- $ F $ is the filter size (assumed square)
- $ S $ is the stride
- $ P $ is the padding (usually 0 for no padding)

### Example:
For an image of size $ 32 \times 32 $, filter size $ 3 \times 3 $, stride $ 1 $, and padding $ 1 $:
$$ O_W = \left\lfloor \frac{32 - 3 + 2 \cdot 1}{1} \right\rfloor + 1 = 32 $$

So, the output size remains $ 32 \times 32 $ after convolution.

## Important Facts about CNNs:
1. **Parameter Efficiency**: CNNs are highly parameter-efficient due to weight sharing in convolution layers.
2. **Translation Invariance**: CNNs can detect objects regardless of their location in the image.
3. **Hierarchical Feature Learning**: Lower layers learn basic features (edges, corners), and higher layers learn more complex features (shapes, objects).
4. **Use of Large Datasets**: CNNs perform best when trained on large labeled datasets.
5. **Pretrained Models**: Pretrained models can be fine-tuned for specific tasks with limited data.

---

# CNN Architectures

## 1. AlexNet:
- **Description**: AlexNet, designed by Geoffrey Hinton's group, is a deep CNN architecture that consists of 5 convolutional layers followed by 3 fully connected layers. It uses ReLU activation functions and a large amount of training data, leading to success in the ImageNet competition in 2012.
- **Key Features**:
  - Use of ReLU for non-linearity.
  - Dropout regularization to prevent overfitting.
  - Max-pooling to reduce spatial dimensions.
  
## 2. GoogLeNet (Inception v1):
- **Description**: GoogLeNet introduced the "Inception module" that utilizes multiple types of filters in parallel, allowing the network to capture multi-scale information. It also uses global average pooling to reduce the number of parameters in the fully connected layer.
- **Key Features**:
  - Inception modules (convolution layers with different kernel sizes).
  - Reduced parameters with global average pooling.
  - 22 layers deep.

## 3. ResNet:
- **Description**: ResNet (Residual Networks) introduces skip connections (residual blocks), which allow the model to skip certain layers, thus mitigating the vanishing gradient problem and enabling very deep networks.
- **Key Features**:
  - Skip connections (residual blocks) for gradient flow.
  - Deep architectures, up to 1000 layers.
  - He initialization for better weight scaling.

## 4. RNN (Recurrent Neural Networks):
- **Description**: RNNs are designed for sequential data, such as time series or natural language. They have loops that allow information to persist from one step to another.
- **Key Features**:
  - Effective for sequence prediction tasks (e.g., time series forecasting, language modeling).
  - Use of hidden states to carry information through time steps.
  - Susceptible to vanishing/exploding gradients, leading to the use of LSTMs and GRUs.

## 5. Transformers:
- **Description**: Transformers are an architecture based on self-attention mechanisms that handle long-range dependencies in sequence data. They have become the backbone of NLP models like GPT and BERT.
- **Key Features**:
  - Self-attention mechanism that allows the model to weigh the importance of each word in a sentence.
  - Highly parallelizable, unlike RNNs.
  - Can be used in both sequence-to-sequence and sequence classification tasks.
  - The transformer model consists of layers of attention and feed-forward networks.
