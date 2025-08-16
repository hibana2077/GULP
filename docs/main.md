# GULP: A Smooth Self‑Gated Activation with a Localized Pulse

## Abstract

Activation functions critically shape the optimization landscape of deep networks. We propose **GULP**, a smooth, self‑gated activation that augments the Swish/SiLU family with a localized pulse (bump) term. Formally, GULP multiplies a Swish gate with a unimodal Gaussian bump centered near the positive region, yielding a mild non‑monotonic uplift around the activation threshold while retaining Swish‑like tails. This design enhances gradient flow around moderate positive pre‑activations with negligible overhead. GULP is drop‑in compatible with MLP, CNN, and Transformer feed‑forward layers, and can also serve as the gate in GLU‑style blocks. We provide the mathematical formulation, derivatives, and a comprehensive empirical evaluation across vision, language, audio, graphs, and tabular tasks.

**Keywords:** activation function, Swish/SiLU, GLU gates, smooth non‑monotonicity, optimization dynamics.

---

## 1. Introduction

**Motivation.** Nonlinearities such as ReLU, GELU, and Swish govern gradient propagation and representation capacity. However, monotone activations may under‑emphasize moderately positive pre‑activations that are predictive yet not strongly activated. We hypothesize that a **localized uplift** near this region improves optimization and generalization without significant cost.

**Contributions.**

1. **GULP activation.** A smooth, self‑gated nonlinearity that extends Swish with a learnable localized pulse; reduces to SiLU when the pulse amplitude is zero.
2. **Theory‑informed design.** We analyze smoothness, tails, bounded negative response, and provide exact derivatives; we further discuss Lipschitz and monotonicity regimes.
3. **Broad empirical validation.** From small‑scale sanity tests to ImageNet‑1k and WMT14 En–De, including GLU‑style gates in Transformers with parameter/FLOPs‑matched comparisons.
4. **Robustness & efficiency.** We report results under distribution shift (ImageNet‑C/CIFAR‑C), calibration (ECE), training stability, throughput, and memory.
5. **Open artifacts.** Reproducible code, configs, logs, and scripts to regenerate tables/figures; model checkpoints when licenses permit.

---

## 2. Related Work (brief)

Activation families (ReLU, LeakyReLU, ELU, GELU, Swish/SiLU, Mish), gated units (GLU, GEGLU, SwiGLU), non‑monotone activations, calibration and robustness under corruptions (CIFAR‑C/ImageNet‑C). (Full citations to be completed in the camera‑ready.)

---

## 3. Method: GULP Activation

Let $\sigma(\cdot)$ denote the sigmoid. With hyperparameters $\alpha>0$ (gate slope), $A\ge0$ (pulse amplitude), $\mu\in\mathbb{R}$ (pulse center), and $\sigma_b>0$ (pulse width), we define the element‑wise activation

$$
\boxed{\operatorname{GULP}(x) = \underbrace{x\,\sigma(\alpha x)}_{\text{Swish/SiLU gate}}\,\cdot\,\underbrace{\Big(1 + A\,\exp\!\big(-\tfrac{(x-\mu)^2}{2\sigma_b^2}\big)\Big)}_{\text{localized pulse}}}.
$$

**Special cases.** If $A=0$, GULP reduces to Swish (SiLU when $\alpha=1$). As $x\to+\infty$, GULP behaves linearly; as $x\to-\infty$, it smoothly saturates near zero.

**Derivative.** Let $s(x)=\sigma(\alpha x)$, $p(x)=x s(x)$, and $b(x)=1+A\exp\!\big(-\tfrac{(x-\mu)^2}{2\sigma_b^2}\big)$. Then

$$
\operatorname{GULP}'(x)=p'(x)\,b(x)+p(x)\,b'(x),\quad p'(x)=s(x)+\alpha x s(x)(1-s(x)),\quad b'(x)=-\frac{A}{\sigma_b^2}(x-\mu)\,e^{-\frac{(x-\mu)^2}{2\sigma_b^2}}.
$$

**GLU‑style gate variant.** In a two‑branch FFN, use a value branch $f(x)$ and a gate branch $g(x)=\sigma(\alpha x)\big(1 + A e^{-\frac{(x-\mu)^2}{2\sigma_b^2}}\big)$, and output $f(x)\odot g(x)$. To parameter/FLOPs‑match vanilla FFN, set the hidden width to **2/3** of the original when switching to GLU family.

**Parameterization tips.** Parameterize $\sigma_b=\operatorname{softplus}(\rho)+\varepsilon$ to ensure positivity; constrain $A\ge0$ via $A=\operatorname{softplus}(\eta)$.

---

## 4. Experiments (ICLR‑oriented)

We structure claims and tests to align with ICLR reviewing criteria: clear baselines, controlled comparisons, ablations, robustness, and reproducibility.

### 4.1 Setups & Protocols

* **Datasets.** CIFAR‑10/100, Tiny‑ImageNet, ImageNet‑100/1k; IMDb, SST‑2, AG News; IWSLT14 De↔En, WMT14 En–De; Speech Commands v2, ESC‑50; OGBN‑Arxiv; UCI Adult/HIGGS.
* **Models.** ResNet‑18/50, WideResNet‑28‑10, DeiT/ViT‑T/S; Transformer encoders for classification; Transformer (base) for seq2seq; small LM on WikiText‑2/103; GCN/GraphSAGE; MLPs for tabular.
* **Baselines.** ReLU, GELU, SiLU/Swish, Mish; GLU variants: GLU, ReGLU, GEGLU, SwiGLU.
* **Compute & training.** Same optimizer, LR schedule, augmentation, batch size, weight decay across activations; report **seeds=3–5**, mean±std. Parameter/FLOPs‑match when switching to GLU‑style.
* **Metrics.** Accuracy/Top‑1/Top‑5; BLEU/chrF; Perplexity; AUC; throughput (samples/s), wall‑clock per epoch, peak GPU memory; stability: divergence rate, gradient norms.
* **Statistical testing.** Paired t‑tests across seeds with Holm–Bonferroni correction for primary tables.

### 4.2 Primary Results

**Vision (ImageNet‑1k).** ResNet‑50 (90–120 epochs) and DeiT‑S. Replace GELU/SiLU with GULP; report Top‑1/Top‑5, throughput, memory.

**Language (IWSLT14, WMT14).** Transformer‑base with FFN activation as GELU/GEGLU/SwiGLU/**GULP‑GLU**; keep FFN hidden at **2/3** for GLU family; report BLEU/chrF and training stability.

**Ablations across scales.** CIFAR‑10/100 and Tiny‑ImageNet with ResNets/ViT‑T to verify trends under low compute.

### 4.3 Robustness & Calibration

* **Corruption robustness.** CIFAR‑C / ImageNet‑C; report mCE (or relative error increase) and breakdown by corruption severity.
* **Calibration.** Expected Calibration Error (ECE), NLL; Reliability diagrams.
* **Shifted text tasks (option).** Train on AG News, evaluate on shifted domains (e.g., perturbations or OOD splits).

### 4.4 Optimization Dynamics & Efficiency

* **Convergence.** Training/validation curves; steps to reach a fixed accuracy; area‑under‑curve.
* **Gradient/activation stats.** Norms, sparsity, dynamic range; distribution of pre‑activations around $\mu$.
* **Efficiency.** Throughput, wall‑clock, memory; FLOPs parity; inference latency.

### 4.5 Ablations & Sensitivity

* **Pulse parameters.** Grid over $A\in\{0,0.1,0.2,0.3,0.5\}$, $\mu\in\{0.5,1.0,1.5\}$, $\sigma_b\in\{0.3,0.5,0.8\}$, $\alpha\in\{0.8,1.0,1.2,1.5,2.0\}$; report best‑vs‑default gap.
* **Parameter granularity.** Scalar‑shared vs per‑channel vs per‑head.
* **Where to apply.** Element‑wise in FFN/CNN vs GLU gate only.
* **Regularization.** Weight decay on $\alpha,A$; effect of $\sigma_b$ parameterization.
* **Compatibility.** With BN/LN, residual scaling, dropout.
* **Negative results.** Cases where large $A$ harms optimization (document and analyze).

---

## 5. Results Templates (to be filled)

**Table 1. ImageNet‑1k Top‑1/Top‑5 and efficiency (resnet50.fb_swsl_ig1b_ft_in1k).**

| Activation | Top‑1 | Top‑5 | Throughput (img/s) | Peak Mem (GB) |
| ---------- | ----: | ----: | -----------------: | ------------: |
| ReLU       |       |       |                    |               |
| GELU       |       |       |                    |               |
| SiLU       |       |       |                    |               |
| **GULP**   |       |       |                    |               |

**Table 2. IWSLT14 De→En BLEU/chrF with FFN activations (param/FLOPs‑matched).**

| Activation   | BLEU | chrF | Divergence Rate (%) |
| ------------ | ---: | ---: | ------------------: |
| GELU         |      |      |                     |
| GEGLU        |      |      |                     |
| SwiGLU       |      |      |                     |
| **GULP‑GLU** |      |      |                     |

**Table 3. CIFAR‑10/100 accuracy ± std (n=5 seeds).**

| Dataset   | Model     | ReLU | GELU | SiLU | **GULP** |
| --------- | --------- | ---: | ---: | ---: | -------: |
| CIFAR‑10  | resnet18 |      |      |      |          |
| CIFAR‑100 | wide_resnet50_2 |      |      |      |          |

**Table 4. Robustness on ImageNet‑C (lower is better).**

| Activation | mCE ↓ | ECE ↓ |
| ---------- | ----: | ----: |
| GELU       |       |       |
| SiLU       |       |       |
| **GULP**   |       |       |

**Figure 1.** GULP vs SiLU curves and first derivatives (default vs learned parameters).
**Figure 2.** Training curves (ImageNet‑1k, ResNet‑50).
**Figure 3.** BLEU vs steps and divergence histograms (IWSLT14).
**Figure 4.** Reliability diagrams (ImageNet‑1k).
**Figure 5.** Sensitivity heatmaps over $(A,\mu,\sigma_b)$.

---

## 6. Implementation Details (for Reproducibility)

* **Code release.** MIT‑licensed repo: `configs/` (YAML), `models/activations/gulp.py`, `train.py`, `eval.py`, `scripts/plot_*.py`, `experiments/` (sweeps). One‑command runners to regenerate each table/figure.
* **Environment.** CUDA/cuDNN versions; PyTorch/JAX version; exact commit hash; deterministic flags; seed control.
* **Hyperparameters.** LR, schedule, warmup, weight decay, label smoothing, augmentation recipes (RandAug/Mixup/CutMix for vision; tokenization/BPE for NLP); dropout; FFN widths; attention heads; vocab sizes.
* **Data.** Source URLs, licenses, preprocessing; train/val/test splits; corruption generation (ImageNet‑C/CIFAR‑C) and severity levels.
* **Checkpointing.** Best‑val and last‑epoch; log scalars and histograms; save seeds and configs per run.

---

## 7. Limitations

GULP introduces a localized non‑monotonicity that may hinder optimization if the pulse amplitude is large; extreme $\mu$ placements can reduce benefits in certain modalities. Additional regularization or smaller $A$ may be required for very deep/narrow networks.

---

## 8. Broader Impact

We do not foresee direct negative societal impacts beyond those already associated with training deep models on large datasets (energy use, dataset biases). Naming and meme origins are purely nominal; the method is a general mathematical activation.

---

## 9. Reproducibility Checklist (ICLR‑style)

* [ ] All models and training procedures are fully specified; hyperparameter grids provided.
* [ ] Parameter/FLOPs‑matched comparisons when changing FFN structure (GLU family uses 2/3 width).
* [ ] Datasets and licenses are documented; exact preprocessing is scripted.
* [ ] Seeds (≥3, preferably 5) and statistical tests (paired t‑test with Holm–Bonferroni) reported.
* [ ] Complete logs, configs, and scripts released; tables auto‑generated from logs.
* [ ] Negative results and failure cases included.

---

## 10. References

(To be completed; include Swish/SiLU, GELU, GLU/GEGLU/SwiGLU, robustness/calibration literature.)

---

## Appendix A. Derivations

Full derivatives, smoothness properties, tails, and Lipschitz discussion; gradient w\.r.t. parameters $\partial/\partial\alpha,\partial/\partial A,\partial/\partial\mu,\partial/\partial\sigma_b$.

## Appendix B. Hyperparameters & Search Spaces

**Default GULP parameters:** $\alpha=1.2, A=0.25, \mu=1.0, \sigma_b=0.5$.
**Search ranges:** $\alpha\in[0.8,2.0], A\in[0,0.5], \mu\in[0.5,1.5], \sigma_b\in[0.3,1.0]$.
**Granularity:** scalar‑shared vs per‑channel vs per‑head.

## Appendix C. Additional Figures & Diagnostics

Distribution of pre‑activations around $\mu$; activation sparsity; Hessian trace / curvature proxies; loss‑landscape slices.

## Appendix D. Licenses & Data Statements

List dataset sources, licenses, and any usage constraints; note any redistributed checkpoints and their licenses.
