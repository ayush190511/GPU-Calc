import React, { useState } from 'react';
import { ChevronDown, Sparkles, Terminal, Cpu, Database, Flame, Scale } from 'lucide-react';

interface FaqItem {
  id: string;
  question: string;
  category: string;
  icon: React.ComponentType<{ className?: string }>;
  answer: React.ReactNode;
}

const FAQS: FaqItem[] = [
  {
    id: 'calculate-kv-cache',
    question: 'How do I calculate KV Cache and activation memory in transformers?',
    category: 'Architecture Math',
    icon: Database,
    answer: (
      <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
        <p>
          In autoregressive transformer generation, key-value tensors for previous tokens are cached to prevent recomputing attention. The mathematical formula is:
        </p>
        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-cyan-300 text-xs overflow-x-auto">
          KV_Cache (Bytes) = 2 (Key & Value) × n_layers × n_kv_heads × head_dim × context_length × batch_size × bytes_per_element
        </div>
        <p>
          With standard <b>FP16 (2 bytes/param)</b> on Llama 3 70B (80 layers, 8 KV heads, 128 head dim), a 128k context window at batch size 1 requires:
        </p>
        <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 font-mono text-slate-200">
          2 × 80 × 8 × 128 × 131,072 × 1 × 2 ≈ 42.9 GB VRAM for KV-cache alone!
        </div>
        <p>
          <b>Optimization:</b> Enabling <b>FP8 KV-cache</b> in vLLM or SGLang cuts this memory footprint by 50% down to ~21.5 GB with zero degradation in perplexity.
        </p>
      </div>
    ),
  },
  {
    id: 'pytorch-oom-backprop',
    question: 'Why does PyTorch throw Out-of-Memory (OOM) errors during backpropagation?',
    category: 'Training & CUDA',
    icon: Flame,
    answer: (
      <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
        <p>
          During the forward pass, PyTorch must store all intermediate tensor activations at every layer to compute gradients via the chain rule during the backward pass.
        </p>
        <ul className="list-disc list-inside space-y-1.5 text-slate-300 ml-1">
          <li><b>Uncheckpointed Activations:</b> Memory scales linearly with <code className="text-amber-300">layers × hidden_dim × sequence_length × batch_size</code>.</li>
          <li><b>AdamW Optimizer States:</b> Standard AdamW tracks 32-bit master weights, 1st momentum (m), and 2nd momentum (v), requiring <b>12 bytes per parameter</b>.</li>
          <li><b>Memory Fragmentation:</b> Dynamic token generation allocates and frees varied tensor blocks, causing PyTorch allocator fragmentation.</li>
        </ul>
        <p>
          <b>Remedy:</b> Enable <b>Gradient Checkpointing</b> (recomputes activations on-the-fly during backward pass to save ~80% activation VRAM) and use <b>8-bit AdamW</b> (bitsandbytes) to cut optimizer states from 12 bytes/param down to 2 bytes/param.
        </p>
      </div>
    ),
  },
  {
    id: 'lora-qlora-vram-savings',
    question: 'How much VRAM does LoRA and QLoRA save compared to full fine-tuning?',
    category: 'Fine-Tuning Comparison',
    icon: Scale,
    answer: (
      <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
        <p>
          <b>QLoRA (Quantized Low-Rank Adaptation)</b> reduces training memory requirements by over <b>90%</b>:
        </p>
        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 p-2">
          <table className="w-full text-left font-mono text-[11px]">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="p-2">Model (70B)</th>
                <th className="p-2">Base Weights</th>
                <th className="p-2">Gradients & Optimizer</th>
                <th className="p-2">Total VRAM Needed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-850">
              <tr>
                <td className="p-2 text-rose-300 font-bold">Full Fine-Tune (FP16)</td>
                <td className="p-2">140 GB</td>
                <td className="p-2">980 GB (AdamW + Grads)</td>
                <td className="p-2 text-rose-400 font-bold">&gt; 1,100 GB (16x A100)</td>
              </tr>
              <tr>
                <td className="p-2 text-cyan-300 font-bold">Standard LoRA (16-bit)</td>
                <td className="p-2">140 GB</td>
                <td className="p-2">~12 GB (r=16 adapters)</td>
                <td className="p-2 text-cyan-400 font-bold">~165 GB (2x A100 80GB)</td>
              </tr>
              <tr>
                <td className="p-2 text-emerald-300 font-bold">QLoRA (4-bit NF4)</td>
                <td className="p-2">38.5 GB</td>
                <td className="p-2">~3.5 GB (r=16 adapters)</td>
                <td className="p-2 text-emerald-400 font-bold">~48 GB (1x A100 or 2x 4090)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          With QLoRA, base weights are frozen in 4-bit NormalFloat precision while gradient updates and optimizer states only apply to low-rank adapter matrices $A$ and $B$.
        </p>
      </div>
    ),
  },
  {
    id: 'cheapest-cloud-gpu',
    question: 'What is the cheapest cloud GPU provider for running open-source LLMs?',
    category: 'Cloud GPU Cost',
    icon: Cpu,
    answer: (
      <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
        <p>
          Specialized GPU clouds offer spot and on-demand pricing up to <b>70–85% cheaper</b> than traditional hyperscalers like AWS or GCP:
        </p>
        <ul className="list-disc list-inside space-y-1.5 text-slate-300 ml-1">
          <li><b>RunPod:</b> RTX 4090 ($0.44/hr spot, $0.69/hr secure), A100 80GB ($1.39/hr spot), H100 SXM5 ($2.49/hr). Best mix of web UI, serverless endpoints, and high reliability.</li>
          <li><b>Vast.ai:</b> Community marketplace with the lowest raw prices (RTX 3090 at $0.17/hr, RTX 4090 at $0.32/hr). Ideal for budget-conscious batch jobs and fine-tuning.</li>
          <li><b>Lambda Labs:</b> Premier datacenter cloud (A100 80GB at $1.69/hr on-demand, H100 at $2.99/hr) with low latency NVLink-4 interconnects.</li>
          <li><b>AWS / GCP:</b> Best for high-compliance enterprise deployments (SOC2, HIPAA, Dedicated VPC), but costs 2x–4x more ($40.97/hr for 8x A100 on AWS p4de).</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'run-70b-on-24gb',
    question: 'Can a 24 GB GPU (RTX 3090 / 4090 / 5090) run a 70B parameter model?',
    category: 'Hardware Sizing',
    icon: Terminal,
    answer: (
      <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
        <p>
          <b>Yes!</b> While an unquantized 70B model requires 140 GB in FP16, applying modern 4-bit and 3-bit quantization techniques allows it to fit onto consumer hardware:
        </p>
        <ul className="list-disc list-inside space-y-1.5 text-slate-300 ml-1">
          <li><b>4-bit AWQ / GPTQ:</b> Compresses the 70B model weights down to <b>~38.5 GB</b>. Running on <b>2x RTX 3090/4090 (48GB combined)</b> gives 50+ tokens/sec over PCIe.</li>
          <li><b>3.5-bit GGUF (Q3_K_M) or EXL2 (3.0 bpw):</b> Compresses weights to <b>~21.5 GB</b>, fitting entirely inside a single <b>24 GB RTX 4090</b> with 4k context at ~25 tokens/sec.</li>
          <li><b>RTX 5090 (32GB GDDR7):</b> Comfortably loads <b>Q4_K_M (4-bit, ~38GB)</b> with partial layer offloading or runs a <b>32B model (Qwen 2.5 32B) in native FP8</b> at blazing 120+ tokens/sec.</li>
        </ul>
      </div>
    ),
  },
];

export const FaqSection: React.FC = () => {
  const [openIds, setOpenIds] = useState<string[]>(['calculate-kv-cache', 'lora-qlora-vram-savings']);

  const toggleOpen = (id: string) => {
    setOpenIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 sm:p-10 backdrop-blur-xl shadow-xl shadow-black/20 space-y-8">
      {/* Header */}
      <div className="max-w-3xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Technical Knowledge Hub</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
          Frequently Asked Questions & Transformer Math
        </h2>
        <p className="text-xs sm:text-sm text-slate-300 mt-2">
          Deep-dive technical answers on KV-cache calculation, CUDA Out-Of-Memory prevention, QLoRA fine-tuning benchmarks, and cloud GPU cost optimization.
        </p>
      </div>

      {/* Accordion List */}
      <div className="space-y-3">
        {FAQS.map((faq) => {
          const isOpen = openIds.includes(faq.id);
          const Icon = faq.icon;
          return (
            <div
              key={faq.id}
              className={`rounded-2xl border transition-all duration-200 overflow-hidden ${
                isOpen
                  ? 'border-indigo-500/40 bg-slate-950/90 shadow-md shadow-indigo-950/30'
                  : 'border-slate-800/80 bg-slate-950/50 hover:border-slate-700'
              }`}
            >
              <button
                type="button"
                onClick={() => toggleOpen(faq.id)}
                className="w-full p-4 sm:p-5 text-left flex items-center justify-between gap-4 cursor-pointer select-none"
                aria-expanded={isOpen}
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center shrink-0 text-indigo-400">
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-400 font-semibold block mb-0.5">
                      {faq.category}
                    </span>
                    <h3 className="text-sm font-bold text-slate-100">{faq.question}</h3>
                  </div>
                </div>

                <div className={`p-1 rounded-lg text-slate-400 transition-transform duration-200 shrink-0 ${isOpen ? 'rotate-180 text-indigo-400' : ''}`}>
                  <ChevronDown className="w-4 h-4" />
                </div>
              </button>

              {isOpen && (
                <div className="px-5 pb-5 pt-1 border-t border-slate-850 animate-fadeIn">
                  {faq.answer}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
