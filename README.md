# GPUCalc — AI Model VRAM & Cloud GPU Cost Estimator 🚀

[![Astro](https://img.shields.io/badge/Astro-7.2-FF5D01.svg?logo=astro&logoColor=white)](https://astro.build)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**GPUCalc** is a high-accuracy, interactive **GPU memory sizing engine and live multi-cloud cost calculator** engineered for AI practitioners, machine learning researchers, and cloud infrastructure architects.

Calculate exact VRAM footprints for **Inference, QLoRA, LoRA, and Full Fine-Tuning**, then instantly compare hardware requirements against live spot and on-demand cloud pricing across **RunPod, Lambda Labs, Vast.ai, AWS EC2, and Google Cloud**.

---

## ✨ Features

- 🧠 **Deterministic VRAM Modeling:** Computes exact model weight memory, Grouped Query Attention (GQA) KV-cache expansion, AdamW optimizer states, gradients, and activation buffers.
- ⚡ **15+ Modern Preloaded Architectures:** Instant presets for **Llama 3.3 (70B), Llama 3.1 (8B), DeepSeek-R1 (671B MoE / 37B active), Qwen 2.5 (72B, 32B, 14B, 7B), Gemma 2 (27B, 9B), Mistral Small 24B, Mixtral 8x7B, Phi-4 (14B)**, and **FLUX.1 Diffusion (12B)**.
- 🎛️ **Multi-Precision & Quantization:** Support for **FP32, FP16/BF16, FP8, INT8, INT4 / AWQ / GPTQ**, and **GGUF (Q4_K_M, Q5_K_M, Q8_0)**.
- 💸 **Live Multi-Cloud Pricing Matrix:** Real-time spot and on-demand pricing across RunPod, Lambda Labs, Vast.ai, AWS EC2 (G5, P4de, P5), GCP (G2, A2, A3), CoreWeave, and Nebius.
- 🚀 **Next-Gen Hardware Index:** Track specs and costs for **NVIDIA Blackwell B200 (192GB), RTX 5090 (32GB), Hopper H200 (141GB), H100 SXM5 (80GB), Ada RTX 4090 / L40S / RTX 6000 Ada**, and **Ampere A100 (40GB/80GB) / RTX 3090**.
- 🌓 **Dual Theme Engine:** Beautiful dark cyberpunk aesthetic and high-contrast light mode with smooth transitions and persistent storage.
- 🔒 **100% Client-Side Privacy:** Zero telemetry. All mathematical evaluations run directly in your local browser session.
- 📋 **One-Click Deploy Generator:** Export CLI launch commands for `vLLM` and HuggingFace `Accelerate` with your selected quantization and context length.

---

## 📐 Mathematical Methodology

### 1. Model Weights Memory
$$M_{\text{weights}} = \frac{\text{Parameters (Billion)} \times 10^9 \times \text{Bytes per Param}}{1024^3}$$

- **FP32:** $4.0 \text{ bytes/param}$
- **FP16 / BF16:** $2.0 \text{ bytes/param}$
- **FP8 / INT8:** $1.0 \text{ byte/param}$
- **INT4 / AWQ / GPTQ:** $\sim 0.55 \text{ bytes/param}$ (includes metadata, scale factor, and zero-point overhead)

### 2. KV-Cache Context Memory (with GQA)
$$M_{\text{kv}} = \frac{2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{Context Length} \times \text{Batch Size} \times \text{Bytes}_{\text{kv}}}{1024^3}$$

Modern architectures (e.g. Llama 3 with 64 query heads and 8 KV heads) utilize **Grouped Query Attention (GQA)**, reducing KV-cache VRAM consumption by up to **87.5%**.

### 3. Training & AdamW Optimizer States
- **Full Fine-Tuning (Mixed Precision):** 
  - FP32 Master Weights: $4 \text{ bytes/param}$
  - AdamW Momentum $m$: $4 \text{ bytes/param}$
  - AdamW Variance $v$: $4 \text{ bytes/param}$
  - Gradients (FP16): $2 \text{ bytes/param}$
  - **Total:** $14 \text{ bytes/param}$
- **QLoRA / LoRA:** Base weights quantized in 4-bit/16-bit; optimizer states and gradients calculated **only for trainable adapter parameters** ($\sim 1.2\%$ of total weights).

### 4. CUDA Runtime & Fragmentation Buffer
$$M_{\text{total}} = (M_{\text{weights}} + M_{\text{kv}} + M_{\text{opt}} + M_{\text{act}} + 1.5\text{GB}) \times 1.15$$

Includes a base **1.5 GB CUDA runtime context** and a **15% headroom buffer** to prevent PyTorch allocator fragmentation and OOM errors during dynamic batching.

---

## 🛠️ Project Structure

```text
vram-calc/
├── public/
│   ├── favicon.svg             # High-contrast GPU vector favicon
│   └── og-image.png            # OpenGraph social banner
├── src/
│   ├── components/
│   │   ├── ThemeToggle.astro   # Animated theme switcher
│   │   └── react/
│   │       ├── GpuCalcApp.tsx  # Master reactive container island
│   │       ├── PricingTable.tsx# Sortable cloud GPU price matrix
│   │       └── VramCalculator.tsx # Interactive parameter sliders & breakdown
│   ├── data/
│   │   └── gpu-pricing.json    # Verified cloud pricing & GPU specs dataset
│   ├── layouts/
│   │   └── BaseLayout.astro    # Global shell, navbar, footer, SEO meta
│   ├── lib/
│   │   ├── calculations.ts     # Core mathematical engine & presets
│   │   └── types.ts            # TypeScript interfaces
│   ├── pages/
│   │   ├── 404.astro           # Diagnostic error page
│   │   ├── about.astro         # Engineering principles & benchmarks
│   │   ├── contact.astro       # Feedback and benchmark submission form
│   │   ├── index.astro         # Main calculator & methodology page
│   │   └── privacy.astro       # Privacy policy & terms of use
│   └── styles/
│       └── global.css          # Tailwind CSS v4 directives & light/dark rules
├── astro.config.mjs            # Astro configuration with React & Tailwind Vite plugin
├── package.json
└── tsconfig.json
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js**: `>= 22.12.0`
- **npm**: `>= 10.0.0`

### Installation
```bash
# Clone the repository
git clone https://github.com/ayush190511/GPU-Calc.git

# Navigate to project directory
cd GPU-Calc

# Install dependencies
npm install
```

### Development
```bash
# Start local dev server at http://localhost:4321
npm run dev
```

### Production Build
```bash
# Build static production bundle into ./dist/
npm run build

# Preview production build locally
npm run preview
```

---

## 🌐 Supported Cloud Providers

| Provider | Supported Instances | Pricing Types |
| :--- | :--- | :--- |
| **RunPod** | RTX 3090, 4090, 5090, L40S, A100, H100 | Spot & Secure Cloud On-Demand |
| **Lambda Labs** | A10G, L40S, A100 (40/80GB), H100, H200 | Spot & On-Demand Cloud |
| **Vast.ai** | RTX 3090, 4090, 5090, Multi-GPU Nodes | Community & Verified Marketplace |
| **AWS EC2** | G5 (A10G), P4de (8x A100 80GB), P5 (8x H100) | Spot & On-Demand Hyperscaler |
| **Google Cloud** | G2 (L4), A2 (A100 80GB), A3 (8x H100) | Preemptible & On-Demand |
| **Nebius / CoreWeave**| H100 SXM5, Blackwell B200 Superclusters | Enterprise & Dedicated Instances |

---

## 🤝 Contributing & Feedback

Contributions, hardware benchmarks, and pricing corrections are welcome!
- Report a bug or pricing discrepancy via [Contact Page](https://github.com/ayush190511/GPU-Calc/issues).
- Submit a Pull Request to add new model presets or cloud providers.

---

## 📄 License

This project is licensed under the **MIT License**.
