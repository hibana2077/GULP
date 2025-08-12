import torch, torch.nn as nn
import matplotlib.pyplot as plt

class GULP(nn.Module):
    def __init__(self, alpha=1.2, bump_amp=0.25, mu=1.0, sigma=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.bump_amp = nn.Parameter(torch.tensor(bump_amp))
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, x):
        swish = x * torch.sigmoid(self.alpha * x)
        bump = 1 + self.bump_amp * torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        return swish * bump

if __name__ == "__main__":
    # 建立模型
    model = GULP()

    # 產生輸入數據
    x = torch.linspace(-5, 5, 500)
    y = model(x)

    # 畫圖
    plt.plot(x.numpy(), y.detach().numpy())
    plt.title("GULP Activation Function")
    plt.xlabel("x")
    plt.ylabel("GULP(x)")
    plt.grid(True)
    plt.savefig("gulp_activation.png")

    
