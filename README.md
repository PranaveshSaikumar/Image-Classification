# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

![image](https://github.com/user-attachments/assets/44c1a85f-dcf3-4dea-abaa-7b2e3fe14f3b)


## DESIGN STEPS
STEP 1: Problem Statement
Define the objective of classifying fashion item images (like T-shirts, trousers, bags, shoes, etc.) using a Convolutional Neural Network (CNN).

STEP 2: Dataset Collection
Use the Fashion-MNIST dataset, which consists of 60,000 training images and 10,000 testing images of 10 different clothing categories. Each image is a 28x28 grayscale image.

STEP 3: Data Preprocessing
- Convert images to tensors.
- Normalize pixel values to the range [0, 1].
- Create DataLoaders for batch processing to enable efficient training and evaluation.

STEP 4: Model Architecture
- Design a Convolutional Neural Network with:
- Multiple convolutional layers followed by ReLU activations and max-pooling.
- Fully connected (dense) layers at the end for classification into 10 fashion categories.

STEP 5: Model Training
- Use CrossEntropyLoss as the loss function.
- Use the Adam optimizer to update the weights.
- Train the model over multiple epochs using the training dataset.

STEP 6: Model Evaluation
- Evaluate model performance on the test dataset.
- Compute classification accuracy.
- Generate and analyze a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization
Save the trained CNN model. Test it on new/unseen images to verify predictions.



## PROGRAM

### Name: Pranavesh Saikumar
### Register Number: 212223040149
```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
      x = self.pool(torch.relu(self.conv1(x)))
      x = self.pool(torch.relu(self.conv2(x)))
      x = self.pool(torch.relu(self.conv3(x)))
      x = x.view(x.size(0),-1)
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)
      return x
```

```
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Pranavesh Saikumar')
        print('Register Number: 212223040149')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/68e56047-d8cc-4ff0-8407-9c04c464665b)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/2e3a9789-9972-4b30-8ad2-e99078ab9067)


### Classification Report

![image](https://github.com/user-attachments/assets/3bbf15da-b77c-4294-bfde-aba729b6e978)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a1badbe2-1e3f-44e9-8f5d-eb45105acd17)

## RESULT
Thus, a convolutional deep neural network for image classification and to verify the response for new images is created successfully.

