import torch
from matplotlib import pyplot as plt


def plot(x_label: int, y_label: int, train_loss_list, output_list, label_list):
    plt.figure(figsize=(x_label, y_label))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.grid(visible=True, which="major", linestyle="-", linewidth=1.5)
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5, linewidth=1.5)

    plt.minorticks_on()
    x1 = torch.arange(0, len(train_loss_list))
    plt_loss = plt.plot(x1, train_loss_list, color="green", label="train loss", linewidth=1, linestyle="-")
    ax1 = plt.gca()
    ax1.set_title("train_loss", fontsize=20)
    ax1.set_xlabel("step", fontsize=20)
    ax1.set_ylabel("loss", fontsize=20)
    plt.tick_params(labelsize=15)

    plt.figure(figsize=(x_label, y_label))
    plt.grid(visible=True, which="major", linestyle="-", linewidth=1.5)
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5, linewidth=1.5)
    plt.minorticks_on()
    x3 = torch.arange(len(output_list)) + 1
    x2 = torch.arange(len(label_list)) + 1
    plt_pred = plt.plot(x3, output_list, color="red", label="prediction", linewidth=2, linestyle="--")
    ax2 = plt.gca()
    ax2.set_title("Prediction on the test dataset(Transformer)", fontsize=22)
    ax2.set_xlabel("Cycle", fontsize=25)
    ax2.set_ylabel("Capacity(mAh/g)", fontsize=25)
    print(len(output_list))
    plt_data = plt.plot(x2, label_list, color="dodgerblue", label="data-01", linewidth=2, linestyle="-")
    plt.tick_params(labelsize=15)
    plt.legend(prop={"size": 20})
    plt.show()


