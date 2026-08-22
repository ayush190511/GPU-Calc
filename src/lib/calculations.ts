export type QuantizationType =
  | 'fp32'
  | 'fp16'
  | 'bf16'
  | 'fp8'
  | 'int8'
  | 'int4'
  | 'gguf_q4_k_m'
  | 'gguf_q5_k_m'
  | 'gguf_q8_0';

export type RunMode =
  | 'inference'
  | 'qlora'
  | 'lora_16bit'
  | 'full_finetune_16bit'
  | 'full_finetune_32bit';

export type KvCacheQuantization = 'fp16' | 'fp8' | 'int4';

export interface ModelPreset {
  id: string;
  name: string;
  family: string;
  parametersB: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  hiddenDim: number;
  defaultContext: number;
  maxContext: number;
  isMoe?: boolean;
  totalMoeParamsB?: number;
  activeMoeParamsB?: number;
  description: string;
}

export interface VramConfig {
  parametersB: number;
  quantization: QuantizationType;
  contextLength: number;
  batchSize: number;
  mode: RunMode;
  kvCacheQuantization: KvCacheQuantization;
  loraRank?: number;
  loraTrainableFraction?: number; // e.g. 0.01 for 1%
  gradientCheckpointing?: boolean;
  cudaOverheadBufferGb?: number; // default ~1.5GB
  safetyMarginPct?: number; // default ~15%
  // Optional architecture overrides:
  layers?: number;
  heads?: number;
  kvHeads?: number;
  headDim?: number;
  hiddenDim?: number;
}

export interface VramBreakdown {
  modelWeightsGb: number;
  kvCacheGb: number;
  optimizerGb: number;
  gradientsGb: number;
  activationsGb: number;
  cudaOverheadGb: number;
  safetyMarginGb: number;
  totalVramGb: number;
  recommendedVramGb: number;
  // Metadata for UI insights
  bytesPerWeightParam: number;
  bytesPerKvParam: number;
  contextLength: number;
  batchSize: number;
  mode: RunMode;
  trainableParamsB: number;
  suggestedGpuCount: {
    rtx4090: number; // 24GB
    rtx5090: number; // 32GB
    a100_80gb: number; // 80GB
    h100_80gb: number; // 80GB
    b200_192gb: number; // 192GB
  };
}

export const QUANT_BYTES: Record<QuantizationType, { bytes: number; label: string; desc: string }> = {
  fp32: { bytes: 4.0, label: 'FP32 (32-bit Float)', desc: 'Standard single precision. 4.0 bytes/param.' },
  fp16: { bytes: 2.0, label: 'FP16 (16-bit Float)', desc: 'Half precision. 2.0 bytes/param.' },
  bf16: { bytes: 2.0, label: 'BF16 (Bfloat16)', desc: 'Brain Floating Point. 2.0 bytes/param. Preferred for modern LLMs.' },
  fp8: { bytes: 1.0, label: 'FP8 (8-bit Float)', desc: 'Modern 8-bit float. 1.0 byte/param. Minimal loss on Ada/Hopper.' },
  int8: { bytes: 1.0, label: 'INT8 (8-bit Integer)', desc: 'Quantized integer. 1.0 byte/param.' },
  int4: { bytes: 0.55, label: 'INT4 / AWQ / GPTQ', desc: '4-bit quant with ~10% scale/zero overhead. ~0.55 bytes/param.' },
  gguf_q4_k_m: { bytes: 0.58, label: 'GGUF Q4_K_M', desc: 'Medium 4-bit k-quant. ~0.58 bytes/param.' },
  gguf_q5_k_m: { bytes: 0.70, label: 'GGUF Q5_K_M', desc: 'Medium 5-bit k-quant. ~0.70 bytes/param.' },
  gguf_q8_0: { bytes: 1.05, label: 'GGUF Q8_0', desc: '8-bit GGUF quant. ~1.05 bytes/param.' },
};

export const KV_QUANT_BYTES: Record<KvCacheQuantization, { bytes: number; label: string }> = {
  fp16: { bytes: 2.0, label: 'FP16 (2.0 B/param)' },
  fp8: { bytes: 1.0, label: 'FP8 (1.0 B/param - vLLM / SGLang)' },
  int4: { bytes: 0.5, label: 'INT4 (0.5 B/param - FlashInfer)' },
};

export const PRESET_MODELS: ModelPreset[] = [
  {
    id: 'llama-3.3-70b',
    name: 'Llama 3.3 70B Instruct',
    family: 'Meta Llama',
    parametersB: 70.6,
    layers: 80,
    heads: 64,
    kvHeads: 8, // GQA 8:1
    headDim: 128,
    hiddenDim: 8192,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'Meta flagship open-weights dense powerhouse with 128k context & GQA.',
  },
  {
    id: 'llama-3.1-8b',
    name: 'Llama 3.1 8B Instruct',
    family: 'Meta Llama',
    parametersB: 8.03,
    layers: 32,
    heads: 32,
    kvHeads: 8, // GQA 4:1
    headDim: 128,
    hiddenDim: 4096,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'Gold standard compact LLM for local reasoning, agents, and fine-tuning.',
  },
  {
    id: 'deepseek-r1-671b',
    name: 'DeepSeek R1 / V3 (671B MoE)',
    family: 'DeepSeek',
    parametersB: 671,
    layers: 61,
    heads: 128,
    kvHeads: 128, // Multi-head Latent Attention (MLA) compressed effective dim ~512
    headDim: 128,
    hiddenDim: 7168,
    defaultContext: 8192,
    maxContext: 131072,
    isMoe: true,
    totalMoeParamsB: 671,
    activeMoeParamsB: 37,
    description: 'Reasoning model with 671B total weights, 37B active parameters, and MLA.',
  },
  {
    id: 'deepseek-r1-distill-70b',
    name: 'DeepSeek R1 Distill 70B (Llama)',
    family: 'DeepSeek',
    parametersB: 70.6,
    layers: 80,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 8192,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'DeepSeek reasoning traces distilled into Llama 3.3 70B architecture.',
  },
  {
    id: 'deepseek-r1-distill-32b',
    name: 'DeepSeek R1 Distill 32B (Qwen)',
    family: 'DeepSeek',
    parametersB: 32.5,
    layers: 64,
    heads: 40,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 5120,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'SOTA mid-weight reasoning distilled into Qwen 2.5 32B.',
  },
  {
    id: 'deepseek-r1-distill-14b',
    name: 'DeepSeek R1 Distill 14B (Qwen)',
    family: 'DeepSeek',
    parametersB: 14.7,
    layers: 48,
    heads: 40,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 5120,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'Top-tier code and math reasoning fitting easily onto a single 24GB/32GB GPU.',
  },
  {
    id: 'qwen-2.5-72b',
    name: 'Qwen 2.5 72B Instruct',
    family: 'Alibaba Qwen',
    parametersB: 72.7,
    layers: 80,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 8192,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'Industry-leading multilingual coding and reasoning dense model.',
  },
  {
    id: 'qwen-2.5-32b',
    name: 'Qwen 2.5 32B Instruct',
    family: 'Alibaba Qwen',
    parametersB: 32.5,
    layers: 64,
    heads: 40,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 5120,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'Sweet-spot balance of 70B performance with 32B memory footprint.',
  },
  {
    id: 'qwen-2.5-7b',
    name: 'Qwen 2.5 7B Instruct',
    family: 'Alibaba Qwen',
    parametersB: 7.61,
    layers: 28,
    heads: 28,
    kvHeads: 4,
    headDim: 128,
    hiddenDim: 3584,
    defaultContext: 8192,
    maxContext: 131072,
    description: 'Super-efficient 7B model for edge deployments and high-throughput pipelines.',
  },
  {
    id: 'mistral-small-24b',
    name: 'Mistral Small 3 24B',
    family: 'Mistral AI',
    parametersB: 24.0,
    layers: 56,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 4096,
    defaultContext: 8192,
    maxContext: 32768,
    description: 'High-capability 24B enterprise reasoning model with fast inference.',
  },
  {
    id: 'mixtral-8x7b',
    name: 'Mixtral 8x7B MoE',
    family: 'Mistral AI',
    parametersB: 46.7,
    layers: 32,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 4096,
    defaultContext: 8192,
    maxContext: 32768,
    isMoe: true,
    totalMoeParamsB: 46.7,
    activeMoeParamsB: 12.9,
    description: 'Sparse Mixture of Experts with 47B total parameters and 13B active.',
  },
  {
    id: 'gemma-2-27b',
    name: 'Gemma 2 27B',
    family: 'Google Gemma',
    parametersB: 27.2,
    layers: 46,
    heads: 32,
    kvHeads: 16,
    headDim: 128,
    hiddenDim: 4608,
    defaultContext: 8192,
    maxContext: 8192,
    description: 'Google high-efficiency architecture with sliding-window attention.',
  },
  {
    id: 'gemma-2-9b',
    name: 'Gemma 2 9B',
    family: 'Google Gemma',
    parametersB: 9.24,
    layers: 42,
    heads: 16,
    kvHeads: 8,
    headDim: 256,
    hiddenDim: 3584,
    defaultContext: 8192,
    maxContext: 8192,
    description: 'Google lightweight model punching well above its weight class.',
  },
  {
    id: 'phi-4-14b',
    name: 'Phi-4 (14B)',
    family: 'Microsoft Phi',
    parametersB: 14.7,
    layers: 40,
    heads: 40,
    kvHeads: 10,
    headDim: 128,
    hiddenDim: 5120,
    defaultContext: 8192,
    maxContext: 16384,
    description: 'Microsoft state-of-the-art synthetic data trained 14B model.',
  },
  {
    id: 'flux-1-schnell',
    name: 'FLUX.1 [schnell] (12B Diffusion)',
    family: 'Black Forest Labs',
    parametersB: 12.0,
    layers: 38,
    heads: 24,
    kvHeads: 24,
    headDim: 128,
    hiddenDim: 3072,
    defaultContext: 512,
    maxContext: 1024,
    description: '12B parameter Flow Matching Rectified Transformer text-to-image generator.',
  },
];

/**
 * Derives approximate transformer architectural hyperparameters when custom parameter count is specified
 */
export function estimateArchitecture(paramsB: number): {
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  hiddenDim: number;
} {
  if (paramsB <= 2) {
    return { layers: 24, heads: 16, kvHeads: 4, headDim: 64, hiddenDim: 2048 };
  } else if (paramsB <= 4) {
    return { layers: 28, heads: 24, kvHeads: 6, headDim: 128, hiddenDim: 3072 };
  } else if (paramsB <= 9) {
    return { layers: 32, heads: 32, kvHeads: 8, headDim: 128, hiddenDim: 4096 };
  } else if (paramsB <= 16) {
    return { layers: 40, heads: 40, kvHeads: 8, headDim: 128, hiddenDim: 5120 };
  } else if (paramsB <= 36) {
    return { layers: 64, heads: 40, kvHeads: 8, headDim: 128, hiddenDim: 5120 };
  } else if (paramsB <= 80) {
    return { layers: 80, heads: 64, kvHeads: 8, headDim: 128, hiddenDim: 8192 };
  } else if (paramsB <= 150) {
    return { layers: 96, heads: 80, kvHeads: 8, headDim: 128, hiddenDim: 10240 };
  } else {
    // 671B or massive dense/MoE
    return { layers: 61, heads: 128, kvHeads: 16, headDim: 128, hiddenDim: 7168 };
  }
}

/**
 * Calculates KV-Cache VRAM in Gigabytes.
 * Formula: 2 (K & V) * n_layers * n_kv_heads * head_dim * context_length * batch_size * bytes_per_param
 */
export function calculateKvCacheGb(
  layers: number,
  kvHeads: number,
  headDim: number,
  contextLength: number,
  batchSize: number,
  kvBytesPerParam: number
): number {
  // Elements = 2 * layers * kvHeads * headDim * contextLength * batchSize
  const totalElements = 2 * layers * kvHeads * headDim * contextLength * batchSize;
  const totalBytes = totalElements * kvBytesPerParam;
  return totalBytes / (1024 * 1024 * 1024);
}

/**
 * Master VRAM Estimation Engine
 */
export function calculateVramRequirements(config: VramConfig): VramBreakdown {
  const {
    parametersB,
    quantization,
    contextLength,
    batchSize,
    mode,
    kvCacheQuantization = 'fp16',
    loraRank = 16,
    loraTrainableFraction,
    gradientCheckpointing = true,
    cudaOverheadBufferGb = 1.5,
    safetyMarginPct = 0.15,
  } = config;

  // Derive trainable fraction based on loraRank (e.g. rank 16 ≈ 1.2%, rank 64 ≈ 4.8%)
  const effectiveLoraFraction = loraTrainableFraction ?? (loraRank / 16) * 0.012;

  const arch = {
    layers: config.layers ?? estimateArchitecture(parametersB).layers,
    heads: config.heads ?? estimateArchitecture(parametersB).heads,
    kvHeads: config.kvHeads ?? estimateArchitecture(parametersB).kvHeads,
    headDim: config.headDim ?? estimateArchitecture(parametersB).headDim,
    hiddenDim: config.hiddenDim ?? estimateArchitecture(parametersB).hiddenDim,
  };

  const bytesPerWeightParam = QUANT_BYTES[quantization].bytes;
  const bytesPerKvParam = KV_QUANT_BYTES[kvCacheQuantization].bytes;

  // 1. Model Weights Memory (GB)
  // 1 Billion parameters = 10^9 params. (10^9 / 1024^3 ≈ 0.93132257 GB in pure binary GiB)
  const totalParams = parametersB * 1e9;
  const modelWeightsGb = (totalParams * bytesPerWeightParam) / (1024 * 1024 * 1024);

  // 2. KV-Cache Memory (GB)
  let kvCacheGb = 0;
  if (mode === 'inference') {
    kvCacheGb = calculateKvCacheGb(
      arch.layers,
      arch.kvHeads,
      arch.headDim,
      contextLength,
      batchSize,
      bytesPerKvParam
    );
  }

  // 3. Training Overhead: Gradients & Optimizer & Trainable Parameters
  let optimizerGb = 0;
  let gradientsGb = 0;
  let trainableParamsB = 0;

  if (mode === 'inference') {
    optimizerGb = 0;
    gradientsGb = 0;
    trainableParamsB = 0;
  } else if (mode === 'qlora') {
    // QLoRA: Base weights in 4-bit, Adapter parameters
    trainableParamsB = parametersB * effectiveLoraFraction;
    const trainableParamsCount = trainableParamsB * 1e9;

    // Gradients in FP16/BF16 (2 bytes/param) for adapters only
    gradientsGb = (trainableParamsCount * 2) / (1024 * 1024 * 1024);

    // AdamW optimizer for adapters (First & Second moments in FP32 = 8 bytes + FP32 Master weights = 4 bytes = 12 bytes/param)
    optimizerGb = (trainableParamsCount * 12) / (1024 * 1024 * 1024);
  } else if (mode === 'lora_16bit') {
    // Standard LoRA: Base weights in 16-bit, Adapters
    trainableParamsB = parametersB * effectiveLoraFraction;
    const trainableParamsCount = trainableParamsB * 1e9;

    gradientsGb = (trainableParamsCount * 2) / (1024 * 1024 * 1024);
    optimizerGb = (trainableParamsCount * 12) / (1024 * 1024 * 1024);
  } else if (mode === 'full_finetune_16bit') {
    // Full Fine-Tuning 16-bit (Mixed Precision):
    trainableParamsB = parametersB;
    const totalTrainable = parametersB * 1e9;

    // Gradients in FP16/BF16 = 2 bytes/param (or FP32 = 4 bytes)
    gradientsGb = (totalTrainable * 2) / (1024 * 1024 * 1024);

    // AdamW Optimizer States:
    // FP32 Master Weights (4 bytes) + Momentum (4 bytes) + Variance (4 bytes) = 12 bytes/param
    optimizerGb = (totalTrainable * 12) / (1024 * 1024 * 1024);
  } else if (mode === 'full_finetune_32bit') {
    // Full Fine-Tuning 32-bit (Pure FP32):
    trainableParamsB = parametersB;
    const totalTrainable = parametersB * 1e9;

    // Gradients in FP32 = 4 bytes/param
    gradientsGb = (totalTrainable * 4) / (1024 * 1024 * 1024);

    // AdamW: Momentum (4 bytes) + Variance (4 bytes) = 8 bytes/param
    optimizerGb = (totalTrainable * 8) / (1024 * 1024 * 1024);
  }

  // 4. Activation Memory (GB)
  // For inference: small scratch space ~ 0.2 - 1.0 GB depending on context and batch
  // For training: depends heavily on context length, batch size, and whether gradient checkpointing is enabled.
  let activationsGb = 0;
  if (mode === 'inference') {
    // Inference activations: ~ batchSize * contextLength * hiddenDim * 2 bytes * 4 (intermediate buffers)
    const actBytes = batchSize * contextLength * arch.hiddenDim * 2 * 4;
    activationsGb = Math.min(Math.max(actBytes / (1024 * 1024 * 1024), 0.3), 8.0);
  } else {
    // Training activations:
    // With gradient checkpointing: ~ 2 * layers * batchSize * contextLength * hiddenDim * bytes (saving only layer boundaries)
    // Without checkpointing: ~ 10-20x higher
    const factor = gradientCheckpointing ? 2.5 : 16;
    const actBytes = arch.layers * batchSize * contextLength * arch.hiddenDim * factor * 2;
    activationsGb = actBytes / (1024 * 1024 * 1024);
  }

  // 5. CUDA Runtime Overhead & Context
  const cudaOverheadGb = cudaOverheadBufferGb;

  // 6. Subtotal before safety margin
  const subtotalGb =
    modelWeightsGb + kvCacheGb + optimizerGb + gradientsGb + activationsGb + cudaOverheadGb;

  // 7. Safety Margin (15% recommended for fragmentation & peak spikes)
  const safetyMarginGb = subtotalGb * safetyMarginPct;

  const totalVramGb = subtotalGb;
  const recommendedVramGb = subtotalGb + safetyMarginGb;

  // Calculate suggested GPU configurations
  const suggestedGpuCount = {
    rtx4090: Math.ceil(recommendedVramGb / 24),
    rtx5090: Math.ceil(recommendedVramGb / 32),
    a100_80gb: Math.ceil(recommendedVramGb / 80),
    h100_80gb: Math.ceil(recommendedVramGb / 80),
    b200_192gb: Math.ceil(recommendedVramGb / 192),
  };

  return {
    modelWeightsGb: Number(modelWeightsGb.toFixed(2)),
    kvCacheGb: Number(kvCacheGb.toFixed(2)),
    optimizerGb: Number(optimizerGb.toFixed(2)),
    gradientsGb: Number(gradientsGb.toFixed(2)),
    activationsGb: Number(activationsGb.toFixed(2)),
    cudaOverheadGb: Number(cudaOverheadGb.toFixed(2)),
    safetyMarginGb: Number(safetyMarginGb.toFixed(2)),
    totalVramGb: Number(totalVramGb.toFixed(2)),
    recommendedVramGb: Number(recommendedVramGb.toFixed(2)),
    bytesPerWeightParam,
    bytesPerKvParam,
    contextLength,
    batchSize,
    mode,
    trainableParamsB: Number(trainableParamsB.toFixed(2)),
    suggestedGpuCount,
  };
}

/**
 * Calculates estimated token generation throughput (tokens/second)
 * based on memory bandwidth and active memory footprint for batch_size=1 autoregressive decoding.
 */
export function estimateInferenceThroughput(
  modelWeightsGb: number,
  kvCacheGb: number,
  gpuMemoryBandwidthGbps: number
): number {
  if (gpuMemoryBandwidthGbps <= 0) return 0;
  const memoryPerTokenGb = modelWeightsGb + kvCacheGb * 0.05; // weights read every step + slice of KV
  if (memoryPerTokenGb <= 0) return 0;
  // Theoretical max tokens/sec ≈ Memory Bandwidth (GB/s) / Model Memory Footprint (GB) * realistic efficiency (65%)
  const tokensPerSec = (gpuMemoryBandwidthGbps / memoryPerTokenGb) * 0.65;
  return Math.min(Math.max(Number(tokensPerSec.toFixed(1)), 1.0), 350.0);
}
