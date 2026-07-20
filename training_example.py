"""
Simple training example using scalar autodiff. This is a simple example of how to use the Scalar class to perform a simple training loop. The goal is to learn the parameters of a linear model that predicts y from x.
The model is defined as y = w * x + b, where w and b are the parameters to be learned. The loss function is the mean squared error between the predicted and true values of y.
"""
from minitorch.scalar import Scalar
from minitorch.datasets import simple
import random

class SimpleModel:
    """A simple Linear classifier : y = sigmoid(w * x + b)"""

    def __init__(self):
        # Initialize parameters
        self.w = Scalar(random.uniform(-1, 1))
        self.w.requires_grad_(True)
        self.b = Scalar(random.uniform(-1, 1))
        self.b.requires_grad_(True)

    def forward(self, x: float) -> Scalar:
        """Predict probability for input x ."""
        return (self.w * x + self.b).sigmoid()
        
    def parameters(self):
        """returns all trainable parameters"""
        return [self.w, self.b]
        
    def zero_grad(self):
        """reset all gradients to none"""
        for p in self.parameters():
            p.zero_grad_()

def binary_cross_entropy(pred:Scalar, target:float) -> Scalar:
    """
    Compute binary cross entropy loss 

    loss = -[y * log(p) + (1 - y) * log(1 - p)]
    where y is the target and p is the predicted probability
    """
    eps = 1e-7 # Numerical stablility
    #Update data in-place to preserce the computation graph history
    pred.data = max(eps, min(1 - eps, pred.data))

    if target == 1:
        return -pred.log()
    else:
        return -(Scalar(1.0) - pred).log()
    

def train_simple():
    """Train on the simple dataset ."""
    #Generate data
    dataset = simple(100)

    #Create model
    model = SimpleModel()
    learning_rate = 0.5

    for epoch in range(100):
        total_loss = 0.0
        correct = 0

        for (x_coord, y_coord), label in zip(dataset.X, dataset.y):
            #zero gradients
            model.zero_grad() 

            #Forward pass
            pred = model.forward(x_coord)

            #Compute loss
            loss = binary_cross_entropy(pred, label)
            total_loss += loss.data

            #Backward pass
            loss.backward()

            #updata parameters (SGD)
            for param in model.parameters():
                if param.derivative is not None:
                    param.data = param.data - learning_rate * param.derivative

            #Track accuracy
            predicted_label = 1 if pred.data > 0.5 else 0
            if predicted_label == label:
                correct += 1
        
        if epoch % 10 == 0:
            accuracy = correct / dataset.N
            print(f"Epoch {epoch}: Loss = {total_loss/dataset.N : .4f}, Accuracy = {accuracy:.2%}")

    print(f"\n Final parameters: w={model.w.data:.4f},b={model.b.data:.4f}")
    print(f"Expected decision boundary: x = 0.5")
    print(f"Learned boundary: x = {-model.b.data / model.w.data:.4f}")

if __name__ == "__main__":
    train_simple()


        



    