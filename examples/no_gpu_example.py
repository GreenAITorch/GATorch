import sys
sys.path.append("../")
import torch
from torch import nn
from GA import GA

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using {} device".format(device))

def create_network():
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.ReLU()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
        
    network = NeuralNetwork()
    model = network.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return model, loss_fn, optimizer, network
    
def main():
    model, loss_fn, optimizer, network = create_network()

    # Create the profiler object and attach a model to it
    ga_measure = GA()
    # Disable measurements for GPU in case NVML is not supported
    ga_measure.attach_model(model, loss=loss_fn, disable_measurements=["gpu"])

    # Let's try to do a single forward pass and a backward pass
    x = torch.zeros([1, 1, 28, 28]).to(device)
    y = torch.zeros([1, 10]).to(device)
    pred = model(x)

    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # We can print the mean energy measures for training! 
    print(ga_measure.get_mean_measurements())  

if __name__ == "__main__":
    main()
