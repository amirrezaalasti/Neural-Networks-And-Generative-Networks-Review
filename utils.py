import matplotlib.pyplot as plt
import torch

def show_samples(dataset):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    # Loop through each subplot and plot an image
    for i in range(4):
        for j in range(4):
            image, label = dataset[i * 4 + j]  # Get image and label
            image_numpy = image.numpy().squeeze()    # Convert image tensor to numpy array
            axs[i, j].imshow(image_numpy, cmap='gray')  # Plot the image
            axs[i, j].axis('off')  # Turn off axis
            axs[i, j].set_title(f"Label: {label}")  # Set title with label

    plt.tight_layout()  # Adjust layout
    plt.show()  # Show plot

def get_accuracy(data_loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total