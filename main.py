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

class SimpleLLaMANetwork(nn.Module):
    """
    A simple neural network inspired by LLaMA architecture with a custom combined activation layer.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output.
        activation_ratio (dict, optional): Ratio of activations for CombU. Defaults to {"relu": 0.5, "elu": 0.25, "nlrelu": 0.25}.
        seed (int, optional): Seed for reproducibility in CombU. Defaults to None.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation_ratio={"relu": 0.5, "elu": 0.25, "nlrelu": 0.25}, seed=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.combu = CombU(hidden_dim, ratio=activation_ratio, seed=seed)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (mx.Tensor): Input tensor.

        Returns:
            mx.Tensor: Output tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.combu(x)
        x = self.fc2(x)
        return x

def main():
    """
    Example usage of the SimpleLLaMANetwork with CombU activation.
    """
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    mx.random.seed(seed)

    # Define network dimensions
    input_dim, hidden_dim, output_dim = 10, 16, 1

    # Initialize the model
    model = SimpleLLaMANetwork(input_dim, hidden_dim, output_dim, seed=seed)

    # Switch the model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = mx.random.normal(shape=(5, input_dim))

    # Perform forward pass
    output = model(dummy_input)

    # Print the model output
    print("Model output:", output)

    

if __name__ == "__main__":
    main()
