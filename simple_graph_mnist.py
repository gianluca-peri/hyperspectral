import json
import matplotlib.pyplot as plt
import os

# Set readable font sizes
plt.rc('font', size=16)

def load_history(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    run_dir = "mnist/run_0"
    linear_path = os.path.join(run_dir, "spectral_linear", "history.json")
    triadic_path = os.path.join(run_dir, "spectral_triadic", "history.json")

    # Load data
    data_linear = None
    data_triadic = None

    if os.path.exists(linear_path):
        data_linear = load_history(linear_path)
    else:
        print(f"Warning: {linear_path} not found.")

    if os.path.exists(triadic_path):
        data_triadic = load_history(triadic_path)
    else:
        print(f"Warning: {triadic_path} not found.")

    if data_linear is None and data_triadic is None:
        print("No data found to plot.")
        return

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plt.suptitle('Spectral Perceptrons on MNIST')

    # Plot Train Loss
    if data_linear:
        ax1.plot(data_linear['train_loss'], label='Spectral Linear')
    if data_triadic:
        ax1.plot(data_triadic['train_loss'], label='Spectral Triadic')
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Validation Accuracy
    if data_linear:
        ax2.plot(data_linear['val_accuracy'], label='Spectral Linear')
    if data_triadic:
        ax2.plot(data_triadic['val_accuracy'], label='Spectral Triadic')
    
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Save plot
    output_path = os.path.join(run_dir, "simple_graph.mnist.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
