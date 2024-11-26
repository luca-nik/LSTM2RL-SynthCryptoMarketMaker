import matplotlib.pyplot as plt
import numpy as np
import torch

class Plotter:
    def __init__(self, output_directory='images/'):
        self.output_directory = output_directory

    def save_figure(self, targets, predictions, target_label, prediction_label, file_name):
        """
        General method to save a plot comparing actual and predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.plot(targets, label=f"Actual {target_label}", linestyle='--', color='blue')
        plt.plot(predictions, label=f"Predicted {prediction_label}", color='red')
        plt.title(f"Actual vs Predicted {target_label}")
        plt.xlabel("Sample Index")
        plt.ylabel(f"{target_label} Value")
        plt.legend()
        plt.savefig(f'{self.output_directory}{file_name}.png', dpi=400)  # Save with 400 DPI
        plt.close()

    def plot_actual_vs_predicted(self, model, train_loader, device):
        """
        Plot and save the actual vs predicted values for each of the target variables.
        """
        # Set model to evaluation mode (important to deactivate dropout, batch norm, etc.)
        model.eval()

        predictions = []
        targets = []

        # Loop through the train_loader to get the data
        with torch.no_grad():
            for orderbook_batch, target_batch in train_loader:
                orderbook_batch = orderbook_batch.to(device)
                target_batch = target_batch.to(device)

                # Get the predictions from the model
                predicted_values = model.predict(orderbook_batch)

                # Store the predictions and actual targets
                predictions.append(predicted_values.cpu().numpy())
                targets.append(target_batch.cpu().numpy())

        # Convert predictions and targets into a numpy array for easier plotting
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Plot and save each figure
        self.save_figure(targets[:, 0], predictions[:, 0], "Best Ask", "Best Ask", "best_ask")
        self.save_figure(targets[:, 1], predictions[:, 1], "Ask Volume", "Ask Volume", "ask_volume")
        self.save_figure(targets[:, 2], predictions[:, 2], "Best Bid", "Best Bid", "best_bid")
        self.save_figure(targets[:, 3], predictions[:, 3], "Bid Volume", "Bid Volume", "bid_volume")

        print("Figures saved successfully!")

    def plot_loss(self, epoch_losses):
        """
        Plot the training loss curve and save it.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_directory}train.png', dpi=400)  # Save with 400 DPI
        plt.close()
