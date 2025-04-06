# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

#### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

#### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

#### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

#### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

#### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

#### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

#### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.



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

![image](https://github.com/user-attachments/assets/b7b07b0f-fe55-4280-9c0f-904628d221f5)

### Classification Report

![image](https://github.com/user-attachments/assets/1756d140-75cb-4b3a-bd92-11743d1ae4c1)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/c15575f8-f3fb-4e0f-9137-208e899e2d22)

## RESULT
Thus, a convolutional deep neural network for image classification and to verify the response for new images is created successfully.

