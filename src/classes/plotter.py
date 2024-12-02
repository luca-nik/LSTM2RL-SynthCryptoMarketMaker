import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, root_output_directory : str ='images/', features: list[str] = [""]):

        self.root_output_directory = root_output_directory
        self.features = features

    def plot_and_save_figure(self, targets: np.array, predictions: np.array, feature : str, file_name: str, target_directory: str):
        """
        General method to save a plot comparing actual and predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.plot(targets, label=f"Actual {feature}", linestyle='--', color='blue')
        plt.plot(predictions, label=f"Predicted {feature}", color='red', linewidth = 0.5)
        plt.title(f"Actual vs Predicted {feature}")
        plt.xlabel("Sample Index")
        plt.ylabel(f"{feature} Value")
        plt.legend()
        plt.savefig(f'{self.root_output_directory}{target_directory}{file_name}.png', dpi=400)  # Save with 400 DPI
        plt.close()

    def plot_actual_vs_predicted(self, targets: np.array, predictions: np.array, target_directory: str = ''):
        """
        Plot and save the actual vs predicted values in their original scale.
        
        Args:
        """

        # Plot and save each figure
        for i in range(targets.shape[1]):
            self.plot_and_save_figure(targets[:, i], predictions[:, i], self.features[i], self.features[i], target_directory)

        print("Figures saved successfully!")

    def plot_loss(self, epoch_losses: np.array, target_directory: str):
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
        plt.savefig(f'{self.root_output_directory}{target_directory}Loss.png', dpi=400)  # Save with 400 DPI
        plt.close()
