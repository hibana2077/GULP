import os
import itertools
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from .gulp import GULP


def parse_list(s: str, type_fn=float):
    """Parse a comma separated list into a Python list using type_fn."""
    return [type_fn(x) for x in s.split(',') if x.strip()]


def safe_val(v):
    """Convert float to a short safe string for filenames."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).replace('-', 'm').replace('.', 'p')


def plot_activation(model: GULP, x, title: str, save_path: Path):
    y = model(x)
    plt.figure(figsize=(4, 3))
    plt.plot(x.numpy(), y.detach().numpy())
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('GULP(x)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='GULP Activation Parameter Sweep')
    parser.add_argument('--alpha', default='0.5,1.0,1.5', help='逗號分隔的 alpha 列表')
    parser.add_argument('--bump_amp', default='0.0,0.25,0.5', help='逗號分隔的 bump_amp 列表')
    parser.add_argument('--mu', default='0.0,1.0', help='逗號分隔的 mu 列表')
    parser.add_argument('--sigma', default='0.5,1.0', help='逗號分隔的 sigma 列表')
    parser.add_argument('--x-range', nargs=2, type=float, default=[-5.0, 5.0], help='x 起訖值')
    parser.add_argument('--points', type=int, default=500, help='取樣點數')
    parser.add_argument('--out-dir', default='docs/imgs', help='輸出資料夾')
    parser.add_argument('--prefix', default='gulp', help='檔名前綴')
    parser.add_argument('--grid', action='store_true', help='輸出綜合子圖網格')
    parser.add_argument('--no-individual', action='store_true', help='不要輸出單張圖片')
    args = parser.parse_args()

    alphas = parse_list(args.alpha)
    bump_amps = parse_list(args.bump_amp)
    mus = parse_list(args.mu)
    sigmas = parse_list(args.sigma)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.linspace(args.x_range[0], args.x_range[1], args.points)

    combos = list(itertools.product(alphas, bump_amps, mus, sigmas))

    if not args.no_individual:
        for a, b, mu, s in combos:
            model = GULP(alpha=a, bump_amp=b, mu=mu, sigma=s)
            fname = f"{args.prefix}_a{safe_val(a)}_b{safe_val(b)}_mu{safe_val(mu)}_s{safe_val(s)}.png"
            title = f"GULP a={a} b={b} mu={mu} s={s}"
            plot_activation(model, x, title, out_dir / fname)

    if args.grid:
        # Grid ordering: vary fastest over sigma, then mu, bump_amp, alpha
        n = len(combos)
        cols = min(6, max(1, int(n ** 0.5)))
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(cols * 2.4, rows * 2.0))
        for idx, (a, b, mu, s) in enumerate(combos, start=1):
            model = GULP(alpha=a, bump_amp=b, mu=mu, sigma=s)
            y = model(x)
            ax = plt.subplot(rows, cols, idx)
            ax.plot(x.numpy(), y.detach().numpy(), lw=1.0)
            ax.set_title(f"a={a}\nb={b}\nmu={mu}\ns={s}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        plt.tight_layout()
        grid_name = f"{args.prefix}_grid_{len(alphas)}x{len(bump_amps)}x{len(mus)}x{len(sigmas)}.png"
        plt.savefig(out_dir / grid_name, dpi=150)
        plt.close()

    print(f"完成，共產生 {len(combos)} 組參數。輸出位置: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
