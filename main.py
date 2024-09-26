import mlx.core as mx
import mlx.nn as nn
import numpy as np

class CombU(nn.Module):
    """
    Combines multiple activation functions applied to different subsets of the input dimensions.

    Args:
        dim (int): Total number of input dimensions.
        ratio (dict): Dictionary mapping activation names to their respective ratio of dimensions.
                      Example: {"relu": 0.5, "elu": 0.25, "nlrelu": 0.25}
        seed (int, optional): Seed for reproducibility. Defaults to None.
    """
    def __init__(self, dim, ratio={"relu": 0.5, "elu": 0.25, "nlrelu": 0.25}, seed=None):
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.seed = seed

        # Define activation functions
        self.activations = {
            "relu": lambda x: mx.maximum(0, x),
            "elu": lambda x: mx.where(x > 0, x, mx.exp(x) - 1),
            "nlrelu": lambda x: mx.log(1 + mx.maximum(0, x))
        }

        # Validate ratio keys
        for key in self.ratio:
            if key not in self.activations:
                raise ValueError(f"Unsupported activation '{key}'. Supported activations: {list(self.activations.keys())}")

        # Calculate dimensions for each activation
        self.dims = self._calculate_dimensions()

    def _calculate_dimensions(self):
        """
        Calculates and assigns the number of dimensions to each activation function based on the provided ratios.

        Returns:
            dict: Mapping from activation names to their assigned dimension indices.
        """
        if self.seed is not None:
            np.random.seed(self.seed)  # Ensure reproducibility if seed is set

        # Initial dimension assignment based on ratio
        dims = {k: max(1, int(round(v * self.dim))) for k, v in self.ratio.items()}
        total_assigned = sum(dims.values())

        # Adjust dimensions to match the total 'dim'
        while total_assigned != self.dim:
            for k in dims:
                if total_assigned == self.dim:
                    break
                if total_assigned < self.dim:
                    dims[k] += 1
                    total_assigned += 1
                elif dims[k] > 1:
                    dims[k] -= 1
                    total_assigned -= 1

        # Assign unique indices to each activation
        all_indices = np.random.permutation(self.dim)
        dim_assignments = {}
        current = 0
        for act, count in dims.items():
            indices = all_indices[current:current + count]
            dim_assignments[act] = mx.array(indices)
            current += count

        return dim_assignments

    def forward(self, x):
        """
        Applies the combined activation functions to the input tensor.

        Args:
            x (mx.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            mx.Tensor: Activated tensor with the same shape as input.
        """
        activated_x = mx.zeros_like(x)
        for act_name, indices in self.dims.items():
            activated_part = self.activations[act_name](x[:, indices])
            activated_x = activated_x.at[:, indices].set(activated_part)
        return activated_x

class ResidualBlock(nn.Module):
    """
    A residual block consisting of two linear layers with activation and optional dropout.

    Args:
        dim (int): Dimension of the input and output.
        combu_ratio (dict): Activation ratio for the CombU layer.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        seed (int, optional): Seed for reproducibility in CombU. Defaults to None.
    """
    def __init__(self, dim, combu_ratio={"relu": 0.5, "elu": 0.25, "nlrelu": 0.25}, dropout_rate=0.1, seed=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.combu1 = CombU(dim, ratio=combu_ratio, seed=seed)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.combu2 = CombU(dim, ratio=combu_ratio, seed=seed)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (mx.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            mx.Tensor: Output tensor after residual connection.
        """
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.combu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.combu2(out)
        out = self.dropout2(out)
        
        out += residual  # Residual connection
        return out

class AdvancedLLaMANetwork(nn.Module):
    """
    An advanced neural network inspired by LLaMA architecture with multiple layers, residual connections,
    batch normalization, dropout, and a custom combined activation layer.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output.
        num_layers (int): Number of residual blocks.
        activation_ratio (dict, optional): Ratio of activations for CombU. Defaults to {"relu": 0.5, "elu": 0.25, "nlrelu": 0.25}.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        seed (int, optional): Seed for reproducibility in CombU. Defaults to None.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,
                 activation_ratio={"relu": 0.5, "elu": 0.25, "nlrelu": 0.25},
                 dropout_rate=0.1, seed=None):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.input_combu = CombU(hidden_dim, ratio=activation_ratio, seed=seed)
        self.input_dropout = nn.Dropout(dropout_rate)

        # Create multiple residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, combu_ratio=activation_ratio, dropout_rate=dropout_rate, seed=seed)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the advanced LLaMA network.

        Args:
            x (mx.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            mx.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.input_combu(x)
        x = self.input_dropout(x)

        x = self.residual_blocks(x)

        x = self.output_layer(x)
        return x

def generate_synthetic_data(num_samples=1000, input_dim=10, output_dim=1, seed=42):
    """
    Generates a synthetic dataset for regression tasks.

    Args:
        num_samples (int, optional): Number of samples to generate. Defaults to 1000.
        input_dim (int, optional): Dimension of input features. Defaults to 10.
        output_dim (int, optional): Dimension of output. Defaults to 1.
        seed (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Tuple containing input features and target labels as mx.Tensors.
    """
    np.random.seed(seed)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    true_weights = np.random.randn(input_dim, output_dim).astype(np.float32)
    y = X @ true_weights + np.random.randn(num_samples, output_dim).astype(np.float32) * 0.5  # Add noise
    return mx.array(X), mx.array(y)

def train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, device='cpu'):
    """
    Trains the neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        optimizer (mx.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (callable): Loss function.
        X_train (mx.Tensor): Training input features.
        y_train (mx.Tensor): Training target labels.
        X_val (mx.Tensor): Validation input features.
        y_val (mx.Tensor): Validation target labels.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Size of each training batch. Defaults to 32.
        device (str, optional): Device to run the training on ('cpu' or 'gpu'). Defaults to 'cpu'.
    """
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_loss = 0.0
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_X = X_train_shuffled[start:end].to(device)
            batch_y = y_train_shuffled[start:end].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        with mx.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = loss_fn(val_outputs, y_val.to(device)).item()

        print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_loss:.4f} - Validation Loss: {val_loss:.4f}")

def evaluate(model, X_test, y_test, loss_fn, device='cpu'):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained neural network model.
        X_test (mx.Tensor): Test input features.
        y_test (mx.Tensor): Test target labels.
        loss_fn (callable): Loss function.
        device (str, optional): Device to run the evaluation on ('cpu' or 'gpu'). Defaults to 'cpu'.

    Returns:
        float: Test loss.
    """
    model.eval()
    model.to(device)
    with mx.no_grad():
        outputs = model(X_test.to(device))
        test_loss = loss_fn(outputs, y_test.to(device)).item()
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

def save_model(model, filepath):
    """
    Saves the model parameters to a file.

    Args:
        model (nn.Module): The neural network model to save.
        filepath (str): Path to the file where the model will be saved.
    """
    mx.save_parameters(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath, device='cpu'):
    """
    Loads the model parameters from a file.

    Args:
        model (nn.Module): The neural network model to load parameters into.
        filepath (str): Path to the file from which to load the model.
        device (str, optional): Device to load the model onto ('cpu' or 'gpu'). Defaults to 'cpu'.
    """
    mx.load_parameters(model, filepath, device=device)
    print(f"Model loaded from {filepath}")

def main():
    """
    Main function to demonstrate training and evaluating the AdvancedLLaMANetwork.
    """
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    mx.random.seed(seed)

    # Check for GPU availability (assuming 'gpu0' is the first GPU)
    device = 'gpu0' if mx.is_available('gpu') else 'cpu'
    print(f"Using device: {device}")

    # Generate synthetic dataset
    X, y = generate_synthetic_data(num_samples=2000, input_dim=10, output_dim=1, seed=seed)

    # Split into training, validation, and test sets
    train_size = int(0.7 * X.shape[0])
    val_size = int(0.15 * X.shape[0])
    test_size = X.shape[0] - train_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Define network dimensions
    input_dim, hidden_dim, output_dim = 10, 64, 1
    num_layers = 4
    activation_ratio = {"relu": 0.5, "elu": 0.3, "nlrelu": 0.2}
    dropout_rate = 0.2
    epochs = 100
    batch_size = 64
    learning_rate = 0.001

    # Initialize the model
    model = AdvancedLLaMANetwork(input_dim, hidden_dim, output_dim, num_layers=num_layers,
                                 activation_ratio=activation_ratio, dropout_rate=dropout_rate, seed=seed)

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = mx.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val,
          epochs=epochs, batch_size=batch_size, device=device)

    # Evaluate the model on the test set
    print("Evaluating on test set...")
    evaluate(model, X_test, y_test, loss_fn, device=device)

    # Save the trained model
    save_path = "advanced_llama_network.params"
    save_model(model, save_path)

    # Load the model (for demonstration)
    load_model(model, save_path, device=device)

    # Re-evaluate after loading to ensure consistency
    print("Re-evaluating after loading the model...")
    evaluate(model, X_test, y_test, loss_fn, device=device)

if __name__ == "__main__":
    main()
