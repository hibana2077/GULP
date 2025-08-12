import torch, torch.nn as nn
import matplotlib.pyplot as plt

class GULP(nn.Module):
    """
    GULP activation:
        base = x * sigmoid(alpha * x)   (Swish)
        bump = prod_k ( 1 + A_k * exp( - (x - mu_k)^2 / (2 sigma_k^2) ) )
        return base * bump
    """
    def __init__(
        self,
        alpha=1.2,
        n_bumps=1,
        bump_amp=0.25,          # float or list/tuple of length n_bumps
        mu=1.0,                 # if single bump OR fallback center when not equal_spaced
        sigma=0.5,              # float or list
        equal_spaced=False,     # 若為 True 則忽略 mu，使用 [mu_min, mu_max] 等距
        mu_min=-1.0,
        mu_max=1.0
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        # amplitudes
        if isinstance(bump_amp, (list, tuple)):
            assert len(bump_amp) == n_bumps
            amps = torch.tensor(bump_amp, dtype=torch.float32)
        else:
            amps = torch.full((n_bumps,), float(bump_amp))
        self.amps = nn.Parameter(amps)

        # mus
        if equal_spaced:
            mus = torch.linspace(mu_min, mu_max, n_bumps)
        else:
            if isinstance(mu, (list, tuple)):
                assert len(mu) == n_bumps
                mus = torch.tensor(mu, dtype=torch.float32)
            else:
                mus = torch.full((n_bumps,), float(mu))
        self.mus = nn.Parameter(mus)

        # sigmas
        if isinstance(sigma, (list, tuple)):
            assert len(sigma) == n_bumps
            sigmas = torch.tensor(sigma, dtype=torch.float32)
        else:
            sigmas = torch.full((n_bumps,), float(sigma))
        self.sigmas = nn.Parameter(sigmas)

    def forward(self, x):
        # Swish part
        swish = x * torch.sigmoid(self.alpha * x)

        # Multi-bump product
        # x shape (...), expand to (..., 1)
        x_exp = x.unsqueeze(-1)
        gauss = torch.exp(-0.5 * ((x_exp - self.mus) / self.sigmas) ** 2)
        factors = 1 + self.amps * gauss
        bump = factors.prod(dim=-1)
        return swish * bump


if __name__ == "__main__":
    # 範例1: 原本單峰 (相容舊版)
    model1 = GULP()

    # 範例2: 三峰，μ 等距於 [-2, 2]
    model3 = GULP(n_bumps=3, bump_amp=0.3, sigma=0.6, equal_spaced=True, mu_min=-2.0, mu_max=2.0)

    x = torch.linspace(-5, 5, 500)
    y1 = model1(x).detach()
    y3 = model3(x).detach()

    plt.figure()
    plt.plot(x.numpy(), y1.numpy(), label="k=1")
    plt.plot(x.numpy(), y3.numpy(), label="k=3 equal spaced")
    plt.title("GULP Activation (multi-peak)")
    plt.xlabel("x")
    plt.ylabel("GULP(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("gulp_activation_multi.png")
