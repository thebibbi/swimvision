"""LSTM-based trajectory prediction for occlusion handling.

Learns swimming movement patterns to predict underwater hand trajectories.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrajectoryLSTM(nn.Module):
    """LSTM network for trajectory prediction."""

    def __init__(
        self,
        input_dim: int = 2,  # x, y coordinates
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 2,  # predicted x, y
        dropout: float = 0.2,
    ):
        """Initialize LSTM model.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension.
            num_layers: Number of LSTM layers.
            output_dim: Output dimension.
            dropout: Dropout probability.
        """
        super(TrajectoryLSTM, self).__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTM predictor. Install with: pip install torch"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            hidden: Hidden state (optional).

        Returns:
            Output predictions and hidden state.
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Get prediction from last time step
        predictions = self.fc(lstm_out)

        return predictions, hidden

    def predict_sequence(
        self,
        initial_sequence: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Predict future trajectory.

        Args:
            initial_sequence: Initial visible trajectory (seq_len, 2).
            num_steps: Number of future steps to predict.

        Returns:
            Predicted trajectory (num_steps, 2).
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            x = initial_sequence.unsqueeze(0)  # (1, seq_len, 2)

            # Get hidden state from initial sequence
            _, hidden = self.lstm(x)

            # Predict future steps
            predictions = []
            current_input = initial_sequence[-1:].unsqueeze(0)  # Last point

            for _ in range(num_steps):
                # Predict next step
                output, hidden = self.lstm(current_input, hidden)
                next_point = self.fc(output[:, -1:, :])  # (1, 1, 2)

                predictions.append(next_point.squeeze(0))

                # Use prediction as next input
                current_input = next_point

            # Stack predictions
            predictions = torch.cat(predictions, dim=0)  # (num_steps, 2)

            return predictions


class TrajectoryDataset(Dataset):
    """Dataset for training trajectory prediction."""

    def __init__(
        self,
        trajectories: list[np.ndarray],
        sequence_length: int = 10,
        prediction_length: int = 5,
    ):
        """Initialize dataset.

        Args:
            trajectories: List of trajectory arrays (each is Nx2).
            sequence_length: Input sequence length.
            prediction_length: Number of steps to predict.
        """
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # Create training samples
        self.samples = []
        for traj in trajectories:
            if len(traj) < sequence_length + prediction_length:
                continue

            for i in range(len(traj) - sequence_length - prediction_length + 1):
                input_seq = traj[i : i + sequence_length]
                target_seq = traj[i + sequence_length : i + sequence_length + prediction_length]
                self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(target_seq),
        )


class LSTMTrajectoryPredictor:
    """LSTM-based trajectory predictor for occlusion handling."""

    def __init__(
        self,
        sequence_length: int = 10,
        prediction_length: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        """Initialize LSTM predictor.

        Args:
            sequence_length: Length of input sequence.
            prediction_length: Length of prediction.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            learning_rate: Learning rate.
            device: Device to run on (cpu or cuda).
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTM predictor. Install with: pip install torch"
            )

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.device = device

        # Initialize model
        self.model = TrajectoryLSTM(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(
        self,
        trajectories: list[np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """Train the LSTM model.

        Args:
            trajectories: List of training trajectories.
            epochs: Number of training epochs.
            batch_size: Batch size.
            validation_split: Fraction of data for validation.
            verbose: Print training progress.

        Returns:
            Training history dictionary.
        """
        # Create dataset
        dataset = TrajectoryDataset(
            trajectories,
            self.sequence_length,
            self.prediction_length,
        )

        # Split into train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions, _ = self.model(inputs)

                # Calculate loss (only on prediction length)
                pred_future = predictions[:, -self.prediction_length :, :]
                loss = self.criterion(pred_future, targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    predictions, _ = self.model(inputs)
                    pred_future = predictions[:, -self.prediction_length :, :]
                    loss = self.criterion(pred_future, targets)

                    val_loss += loss.item()

            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def predict(
        self,
        visible_trajectory: np.ndarray,
        num_steps: int | None = None,
    ) -> np.ndarray:
        """Predict future trajectory during occlusion.

        Args:
            visible_trajectory: Last visible trajectory points (Nx2).
            num_steps: Number of steps to predict (default: prediction_length).

        Returns:
            Predicted trajectory (num_steps x 2).
        """
        if num_steps is None:
            num_steps = self.prediction_length

        # Ensure we have enough visible points
        if len(visible_trajectory) < self.sequence_length:
            # Pad with the first point if needed
            padding = np.repeat(
                visible_trajectory[0:1],
                self.sequence_length - len(visible_trajectory),
                axis=0,
            )
            visible_trajectory = np.vstack([padding, visible_trajectory])
        else:
            # Use last sequence_length points
            visible_trajectory = visible_trajectory[-self.sequence_length :]

        # Convert to tensor
        input_seq = torch.FloatTensor(visible_trajectory).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict_sequence(input_seq, num_steps)

        return predictions.cpu().numpy()

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model.
        """
        torch.save(  # nosec B614
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "sequence_length": self.sequence_length,
                "prediction_length": self.prediction_length,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )

    def load(self, path: str):
        """Load model from file.

        Args:
            path: Path to model file.
        """
        checkpoint = torch.load(path, map_location=self.device)  # nosec B614

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.sequence_length = checkpoint["sequence_length"]
        self.prediction_length = checkpoint["prediction_length"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])


def create_synthetic_training_data(
    num_trajectories: int = 100,
    num_points: int = 50,
) -> list[np.ndarray]:
    """Create synthetic swimming trajectories for training.

    Args:
        num_trajectories: Number of trajectories to generate.
        num_points: Points per trajectory.

    Returns:
        List of trajectory arrays.
    """
    trajectories = []

    for _ in range(num_trajectories):
        t = np.linspace(0, 2 * np.pi, num_points)

        # Elliptical path with random variations
        a = np.random.uniform(80, 120)  # Width
        b = np.random.uniform(40, 60)  # Height
        phase = np.random.uniform(0, 2 * np.pi)

        x = 200 + a * np.sin(t + phase)
        y = 300 + b * np.cos(t + phase)

        # Add noise
        x += np.random.normal(0, 3, num_points)
        y += np.random.normal(0, 2, num_points)

        trajectory = np.column_stack([x, y])
        trajectories.append(trajectory)

    return trajectories
