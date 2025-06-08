# visualize.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path):
    df = pd.read_csv(csv_path)
    epochs = df['epoch']

    # 绘制损失曲线
    plt.figure()
    plt.plot(epochs, df['train_loss'], label='Train Loss')
    plt.plot(epochs, df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(epochs, df['train_acc'], label='Train Acc')
    plt.plot(epochs, df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.show()

if __name__ == '__main__':
    plot_metrics('checkpoints/metrics.csv')
