import torch


def accuracy(y_pred, y):
    # calculating accuracy
    return (y_pred == y).float().sum() / y_pred.shape[0] * 100


def bce_loss(predictions, target):
    first_half = target * torch.log(predictions + 1e-8)
    second_half = (1 - target) * torch.log(1 - predictions + 1e-8)
    return -torch.mean(first_half + second_half)


def _relu(x):
    return x * (x > 0).float()


def _sigmoid(z):
    return 1 / (1 + torch.exp(-z))


class NeuralNetwork():
    def __init__(self, x):
        num_samples = x.shape[0]
        num_features = x.shape[1]

        self.Z1, self.Z2 = None, None
        self.A1 = None

        self.y_pred = None

        self.w = [
            torch.randn(num_features, 15) * 0.01,       # input layer -> hidden layer
            torch.randn(15, 1) * 0.01                   # hidden layer 1 -> output layer
        ]
        self.b = [
            torch.zeros(15),            # bias for hidden layer 1
            torch.zeros(1)              # bias for output layer
        ]
        self.learning_rate = 0.5

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.values).float()

        # hidden layer 1 -- 15 neurons
        self.Z1 = torch.matmul(x, self.w[0]) + self.b[0]
        self.A1 = _relu(self.Z1)

        # output layer -- 1 neuron
        self.Z2 = torch.matmul(self.A1, self.w[1]) + self.b[1]
        self.y_pred = _sigmoid(self.Z2)

        return self.y_pred

    def backward(self, x, y):
        """
        We have two layers, so we need to calculate 4 gradients:
            1) Gradient for weights and bias in output layer
            2) Gradient for weights and bias in hidden layer 1
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.values).float()

        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y.values).view(-1, 1)

        batch_size = x.shape[0]

        # gradient for weights going into the output layer
        delta2 = self.y_pred - y
        grad_w2 = torch.matmul(self.A1.T, delta2) / batch_size      # batch size normalization
        grad_b2 = torch.mean(delta2, dim=0)     # we do mean and not sum to keep the update size small

        # gradient for the weights going into the hidden layer 1
        delta1 = torch.matmul(delta2, self.w[1].T) * (self.Z1 > 0).float()      # ReLU derivative
        grad_w1 = torch.matmul(x.T, delta1) / batch_size        # batch size normalization
        grad_b1 = torch.mean(delta1, dim=0)

        # updating weights and biases
        self.w[0] -= self.learning_rate * grad_w1
        self.w[1] -= self.learning_rate * grad_w2
        self.b[0] -= self.learning_rate * grad_b1
        self.b[1] -= self.learning_rate * grad_b2
