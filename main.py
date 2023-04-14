import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

input_size = 10
hidden_size = 5
output_size = 1

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(32, input_size)
y = torch.randn(32, output_size)

num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad()

    # Print the loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
