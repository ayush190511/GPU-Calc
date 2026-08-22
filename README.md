# GPUCalc — AI Model VRAM & Cloud GPU Cost Estimator 🚀

[![Astro](https://img.shields.io/badge/Astro-5.0-FF5D01.svg?logo=astro&logoColor=white)](https://astro.build)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**GPUCalc** is a high-accuracy, interactive **GPU memory sizing engine and live multi-cloud cost calculator** engineered for AI practitioners, machine learning researchers, and cloud infrastructure architects.

Calculate exact VRAM footprints for **Inference, QLoRA, LoRA, and Full Fine-Tuning**, configure **Multi-GPU Tensor Parallelism (`1x`, `2x`, `4x`, `8x GPUs`)**, simulate **Vision-Language Model (VLM) Image Tokens**, evaluate **Throughput ROI (Tokens per Dollar)**, and run **On-Premises vs. Cloud TCO (Buy vs. Rent)** break-even comparisons across **RunPod, Lambda Labs, Vast.ai, AWS EC2, and Google Cloud**.

---

## ✨ Features

- 🧠 **Deterministic VRAM Modeling:** Computes exact model weight memory, Grouped Query Attention (GQA) KV-cache expansion, AdamW optimizer states, gradients, and activation buffers.
- 🔀 **Multi-GPU Tensor Parallelism (TP) Sizer:** Seamlessly split workloads across `1x`, `2x`, `4x`, or `8x GPUs`. Computes exact per-card VRAM requirement and alerts on interconnect requirements (PCIe Gen4/5, NVLink-4, or NVLink-5 NVSwitch).
- 👁️ **Vision-Language (VLM) & Multimodal Tokens:** Calculate image patch tokens inside the KV-cache for multimodal frontier models (**Llama 3.2 Vision 11B/90B, Qwen 2.5-VL 7B/72B, Pixtral 12B**) with adjustable resolution (`512x512 = ~400 tokens`, `1024x1024 = ~1,600 tokens`).
- ⚡ **24 Preloaded Modern Architectures:** Instant presets for **DeepSeek-R1 (671B MoE / 37B active), DeepSeek-V3, Llama 3.3 (70B), Llama 3.1 (8B), Qwen 2.5 (72B, 32B, 14B, 7B), Gemma 2 (27B, 9B), Mistral Large 2 (123B), Mistral Small 24B, Mixtral 8x7B, Phi-4 (14B), FLUX.1 [schnell] (12B)**, and **SDXL 1.0 (3.5B)**.
- 🎛️ **Multi-Precision & Quantization:** Support for **FP32, FP16/BF16, FP8, INT8, INT4 / AWQ / GPTQ**, and **GGUF (Q4_K_M, Q5_K_M, Q8_0)**.
- 📊 **Throughput ROI Engine (Tokens per Dollar):** Evaluate **Millions of Tokens per Dollar (MTok / $)** and **Cost per 1M Tokens ($/1M Tok)** for every GPU instance across all 8 cloud providers based on hardware memory bandwidth.
- 🔌 **Buy vs. Rent TCO Simulator:** An interactive on-premise hardware vs. cloud rental simulator. Factors in GPU MSRP, rig build cost, and electricity at $\$0.14/\text{kWh}$ ($450\text{W}$–$600\text{W}$ TDP) to calculate the exact break-even horizon in months and 1-year/2-year cost savings.
- 🔗 **Deep-Linking & Shareable State Permalinks:** 1-click shareable URLs (`?model=llama-3-3-70b&quant=int4&ctx=32768&tp=2`) to share exact hardware configurations with your team.
- 💸 **Live Multi-Cloud Pricing Matrix:** Real-time spot and on-demand pricing across RunPod, Lambda Labs, Vast.ai, AWS EC2 (G5, P4de, P5), GCP (G2, A2, A3), CoreWeave, and Nebius.
- 🚀 **Next-Gen Hardware Index:** Track specs and costs for **NVIDIA Blackwell B200 (192GB), RTX 5090 (32GB), Hopper H200 (141GB), H100 SXM5 (80GB), Ada RTX 4090 / L40S / RTX 6000 Ada**, and **Ampere A100 (40GB/80GB) / RTX 3090**.
- 📋 **Multi-GPU Deploy Generator:** Export CLI launch commands for `vLLM` (with `--tensor-parallel-size`) and HuggingFace `Accelerate` (with `--num_processes`).
- 🌓 **Dual Theme Engine:** Beautiful dark cyberpunk aesthetic and high-contrast light mode with smooth transitions and persistent storage.
- 🔒 **100% Client-Side Privacy:** Zero telemetry. All mathematical evaluations run directly in your local browser session.

---

## 📐 Mathematical Methodology

### 1. Model Weights Memory
$$M_{\text{weights}} = \frac{\text{Parameters (Billion)} \times 10^9 \times \text{Bytes per Param}}{1024^3}$$

- **FP32:** $4.0 \text{ bytes/param}$
- **FP16 / BF16:** $2.0 \text{ bytes/param}$
- **FP8 / INT8:** $1.0 \text{ byte/param}$
- **INT4 / AWQ / GPTQ:** $\sim 0.55 \text{ bytes/param}$ (includes metadata, scale factor, and zero-point overhead)

### 2. KV-Cache Context Memory (with GQA & Vision Tokens)
$$\text{Tokens}_{\text{total}} = \text{Context Length} + (\text{Image Count} \times \text{Tokens per Image})$$
$$M_{\text{kv}} = \frac{2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{Tokens}_{\text{total}} \times \text{Batch Size} \times \text{Bytes}_{\text{kv}}}{1024^3}$$

- **Grouped Query Attention (GQA):** Architectures like Llama 3 (64 query heads : 8 KV heads) reduce KV-cache VRAM by up to **87.5%**.
- **Multimodal Patch Sizing:** $1024 \times 1024 \approx 1,600 \text{ tokens/image}$; $512 \times 512 \approx 400 \text{ tokens/image}$.

### 3. Training & AdamW Optimizer States
- **Full Fine-Tuning (Mixed Precision):** 
  - FP32 Master Weights: $4 \text{ bytes/param}$
  - AdamW Momentum $m$: $4 \text{ bytes/param}$
  - AdamW Variance $v$: $4 \text{ bytes/param}$
  - Gradients (FP16): $2 \text{ bytes/param}$
  - **Total:** $14 \text{ bytes/param}$
- **QLoRA / LoRA:** Base weights quantized in 4-bit/16-bit; optimizer states and gradients calculated **only for trainable adapter parameters** ($\sim 1.2\%$ of total weights).

### 4. Multi-GPU Tensor Parallelism (TP) Split
$$M_{\text{per\_gpu}} = \frac{M_{\text{weights}} + M_{\text{kv}} + M_{\text{opt}} + M_{\text{grad}} + M_{\text{act}}}{\text{TP}} + M_{\text{cuda}}$$

Weights, KV-cache, optimizer states, and activations are sharded evenly across all $\text{TP}$ GPUs, with CUDA runtime overhead allocated per device.

### 5. Throughput ROI & Cost per 1M Tokens
$$\text{Cost per 1M Tokens} = \frac{\text{Hourly Price}}{\text{Tokens/sec} \times 3.6}$$
$$\text{Tokens per Dollar (MTok / \$)} = \frac{\text{Tokens/sec} \times 3600}{\text{Hourly Price} \times 10^6}$$

---

## 🛠️ Project Structure

```text
vram-calc/
├── public/
│   ├── favicon.svg             # High-contrast microchip vector favicon
│   └── og-image.png            # OpenGraph social banner
├── src/
│   ├── components/
│   │   ├── ScrollToTop.astro   # Floating smooth scroll-to-top button
│   │   ├── ThemeToggle.astro   # Animated theme switcher
│   │   └── react/
│   │       ├── FaqSection.tsx  # Interactive FAQ accordion with formula callouts
│   │       ├── GpuCalcApp.tsx  # Master reactive coordinator & URL state permalinks
│   │       ├── PricingTable.tsx# Sortable cloud GPU matrix, ROI engine & TCO lab
│   │       └── VramCalculator.tsx # Sliders, TP cluster selector & VLM tokens
│   ├── data/
│   │   ├── gpu-pricing.json    # 30+ verified cloud pricing & GPU hardware dataset
│   │   └── preset-models.json  # 24 pre-architected LLM & VLM model specs
│   ├── layouts/
│   │   └── BaseLayout.astro    # Global shell, sticky navbar, modern footer & JSON-LD
│   ├── lib/
│   │   ├── calculations.ts     # Core VRAM, TP sharding, ROI & TCO break-even engine
│   │   └── types.ts            # TypeScript interfaces
│   ├── pages/
│   │   ├── 404.astro           # Diagnostic error page
│   │   ├── about.astro         # Engineering principles & benchmarks
│   │   ├── contact.astro       # Feedback and benchmark submission form
│   │   ├── index.astro         # Main calculator & methodology page
│   │   ├── privacy.astro       # Privacy policy & terms of use
│   │   └── calculator/
│   │       └── [model].astro   # 24 programmatic SEO spoke pages with JSON-LD
│   └── styles/
│       └── global.css          # Tailwind CSS v4 directives & light/dark rules
├── astro.config.mjs            # Astro configuration with React & Tailwind Vite plugin
├── package.json
└── tsconfig.json
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js**: `>= 20.0.0` (Recommended: Node 22+)
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

### Diagnostics & Validation
```bash
# Run complete type & template check across all files
npx astro check
```

### Production Build
```bash
# Build static production bundle (29 static pages pre-rendered)
npm run build

# Preview production build locally
npm run preview
```

---

## 🌐 Supported Cloud Providers

| Provider | Supported Instances | Pricing Types |
| :--- | :--- | :--- |
| **RunPod** | RTX 3090, 4090, 5090, L40S, A100 (80GB), H100 SXM5 | Spot & Secure Cloud On-Demand |
| **Lambda Labs** | A10G, L40S, A100 (40/80GB), H100, H200 (141GB) | Spot & On-Demand Cloud |
| **Vast.ai** | RTX 3090, 4090, 5090, Multi-GPU Sharded Nodes | Community & Verified Marketplace |
| **AWS EC2** | G5 (A10G), P4de (8x A100 80GB), P5 (8x H100 SXM5) | Spot & On-Demand Hyperscaler |
| **Google Cloud** | G2 (L4), A2 (A100 80GB), A3 (8x H100 SXM5) | Preemptible & On-Demand |
| **Nebius / CoreWeave**| H100 SXM5, Blackwell B200 Superclusters | Enterprise & Dedicated Instances |

---

## 👨‍💻 Author & Connect

- **Creator:** Ayush
- **GitHub:** [@ayush190511](https://github.com/ayush190511)
- **LinkedIn:** [Ayush on LinkedIn](https://www.linkedin.com/in/ayush190511/)

---

## 🤝 Contributing & Feedback

Contributions, hardware benchmarks, and pricing corrections are welcome!
- Report a bug or pricing discrepancy via [GitHub Issues](https://github.com/ayush190511/GPU-Calc/issues).
- Submit a Pull Request to add new model presets or cloud providers.

---

## 📄 License

This project is licensed under the **MIT License**.
