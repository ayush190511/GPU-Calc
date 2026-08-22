import presetModelsData from '../data/preset-models.json';

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
  slug?: string;
  family: string;
  developer?: string;
  parametersB: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  hiddenDim: number;
  defaultContext: number;
  maxContext: number;
  isMoe?: boolean;
  isVlm?: boolean;
  totalMoeParamsB?: number;
  activeMoeParamsB?: number;
  license?: string;
  description: string;
  recommendedQuant?: string;
  minGpuInference?: string;
  minGpuTraining?: string;
}

export interface VramConfig {
  parametersB: number;
  quantization: QuantizationType;
  contextLength: number;
  batchSize: number;
  mode: RunMode;
  kvCacheQuantization: KvCacheQuantization;
  tensorParallelism?: number; // 1, 2, 4, 8 GPUs
  // Multimodal / VLM options:
  isVlm?: boolean;
  imageCount?: number; // e.g. 0 to 8
  imageResolution?: number; // 512, 1024
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
  // Multi-GPU Tensor Parallelism fields:
  tensorParallelism: number;
  vramPerGpuGb: number;
  recommendedPerGpuVramGb: number;
  interconnectRequirement: string;
  // Multimodal / VLM tokens info:
  vlmImageTokens: number;
  effectiveContextLength: number;
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

export const PRESET_MODELS: ModelPreset[] = presetModelsData as ModelPreset[];

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
    tensorParallelism = 1,
    isVlm = false,
    imageCount = 0,
    imageResolution = 1024,
    loraRank = 16,
    loraTrainableFraction,
    gradientCheckpointing = true,
    cudaOverheadBufferGb = 1.5,
    safetyMarginPct = 0.15,
  } = config;

  // 1. Multimodal / Vision Tokens: 1024x1024 ≈ 1600 tokens/img; 512x512 ≈ 400 tokens/img
  const tokensPerImage = imageResolution === 512 ? 400 : 1600;
  const vlmImageTokens = isVlm && imageCount > 0 ? imageCount * tokensPerImage : 0;
  const effectiveContextLength = contextLength + vlmImageTokens;

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

  // 2. Model Weights Memory (GB)
  const totalParams = parametersB * 1e9;
  const modelWeightsGb = (totalParams * bytesPerWeightParam) / (1024 * 1024 * 1024);

  // 3. KV-Cache Memory (GB) using effectiveContextLength
  let kvCacheGb = 0;
  if (mode === 'inference') {
    kvCacheGb = calculateKvCacheGb(
      arch.layers,
      arch.kvHeads,
      arch.headDim,
      effectiveContextLength,
      batchSize,
      bytesPerKvParam
    );
  }

  // 4. Training Overhead: Gradients & Optimizer & Trainable Parameters
  let optimizerGb = 0;
  let gradientsGb = 0;
  let trainableParamsB = 0;

  if (mode === 'inference') {
    optimizerGb = 0;
    gradientsGb = 0;
    trainableParamsB = 0;
  } else if (mode === 'qlora') {
    trainableParamsB = parametersB * effectiveLoraFraction;
    const trainableParamsCount = trainableParamsB * 1e9;
    gradientsGb = (trainableParamsCount * 2) / (1024 * 1024 * 1024);
    optimizerGb = (trainableParamsCount * 12) / (1024 * 1024 * 1024);
  } else if (mode === 'lora_16bit') {
    trainableParamsB = parametersB * effectiveLoraFraction;
    const trainableParamsCount = trainableParamsB * 1e9;
    gradientsGb = (trainableParamsCount * 2) / (1024 * 1024 * 1024);
    optimizerGb = (trainableParamsCount * 12) / (1024 * 1024 * 1024);
  } else if (mode === 'full_finetune_16bit') {
    trainableParamsB = parametersB;
    const totalTrainable = parametersB * 1e9;
    gradientsGb = (totalTrainable * 2) / (1024 * 1024 * 1024);
    optimizerGb = (totalTrainable * 12) / (1024 * 1024 * 1024);
  } else if (mode === 'full_finetune_32bit') {
    trainableParamsB = parametersB;
    const totalTrainable = parametersB * 1e9;
    gradientsGb = (totalTrainable * 4) / (1024 * 1024 * 1024);
    optimizerGb = (totalTrainable * 8) / (1024 * 1024 * 1024);
  }

  // 5. Activation Memory (GB)
  let activationsGb = 0;
  if (mode === 'inference') {
    const actBytes = batchSize * effectiveContextLength * arch.hiddenDim * 2 * 4;
    activationsGb = Math.min(Math.max(actBytes / (1024 * 1024 * 1024), 0.3), 8.0);
  } else {
    const factor = gradientCheckpointing ? 2.5 : 16;
    const actBytes = arch.layers * batchSize * effectiveContextLength * arch.hiddenDim * factor * 2;
    activationsGb = actBytes / (1024 * 1024 * 1024);
  }

  // 6. CUDA Runtime Overhead & Context
  const cudaOverheadGb = cudaOverheadBufferGb;

  // 7. Subtotal before safety margin
  const subtotalGb =
    modelWeightsGb + kvCacheGb + optimizerGb + gradientsGb + activationsGb + cudaOverheadGb;

  // 8. Safety Margin (15% recommended for fragmentation & peak spikes)
  const safetyMarginGb = subtotalGb * safetyMarginPct;

  const totalVramGb = subtotalGb;
  const recommendedVramGb = subtotalGb + safetyMarginGb;

  // 9. Multi-GPU Tensor Parallelism (TP) Split
  const tp = Math.max(1, tensorParallelism);
  // Weights, KV, Optimizer, and Activations are sharded across GPUs
  const vramPerGpuGb =
    (modelWeightsGb + kvCacheGb + optimizerGb + gradientsGb + activationsGb) / tp + cudaOverheadGb;
  const recommendedPerGpuVramGb = vramPerGpuGb * (1 + safetyMarginPct);

  let interconnectRequirement = 'Single GPU (PCIe Gen4 / Gen5)';
  if (tp === 2) {
    interconnectRequirement = 'PCIe Gen4 / Gen5 or NVLink-4';
  } else if (tp === 4) {
    interconnectRequirement = 'NVLink-4 / NVSwitch (900 GB/s)';
  } else if (tp >= 8) {
    interconnectRequirement = 'NVLink-5 / NVSwitch (1.8 TB/s)';
  }

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
    tensorParallelism: tp,
    vramPerGpuGb: Number(vramPerGpuGb.toFixed(2)),
    recommendedPerGpuVramGb: Number(recommendedPerGpuVramGb.toFixed(2)),
    interconnectRequirement,
    vlmImageTokens,
    effectiveContextLength,
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
 * Convenient calculateVram functional signature
 */
export function calculateVram(
  paramsB: number,
  precision: QuantizationType = 'fp16',
  contextLen: number = 8192,
  batchSize: number = 1,
  mode: RunMode = 'inference',
  isGQA: boolean = true,
  tp: number = 1
): VramBreakdown {
  const arch = estimateArchitecture(paramsB);
  const kvHeads = isGQA ? Math.max(1, Math.floor(arch.heads / 4)) : arch.heads;
  return calculateVramRequirements({
    parametersB: paramsB,
    quantization: precision,
    contextLength: contextLen,
    batchSize: batchSize,
    mode: mode,
    kvCacheQuantization: 'fp16',
    tensorParallelism: tp,
    layers: arch.layers,
    heads: arch.heads,
    kvHeads: kvHeads,
    headDim: arch.headDim,
    hiddenDim: arch.hiddenDim,
  });
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
  const memoryPerTokenGb = modelWeightsGb + kvCacheGb * 0.05;
  if (memoryPerTokenGb <= 0) return 0;
  const tokensPerSec = (gpuMemoryBandwidthGbps / memoryPerTokenGb) * 0.65;
  return Math.min(Math.max(Number(tokensPerSec.toFixed(1)), 1.0), 350.0);
}

/**
 * Calculates Cost per 1 Million Tokens ($/1M Tok)
 */
export function calculateCostPerMillionTokens(
  hourlyPrice: number,
  tokensPerSec: number
): number {
  if (hourlyPrice <= 0 || tokensPerSec <= 0) return 0;
  // 1 hour = 3600 seconds. Total tokens generated in 1 hour = tokensPerSec * 3600
  // Cost per token = hourlyPrice / (tokensPerSec * 3600)
  // Cost per 1M tokens = Cost per token * 1,000,000 = (hourlyPrice / (tokensPerSec * 3.6))
  const cost = hourlyPrice / (tokensPerSec * 3.6);
  return Number(cost.toFixed(4));
}

/**
 * Calculates Tokens Generated Per Dollar (MTok / $)
 */
export function calculateTokensPerDollar(
  hourlyPrice: number,
  tokensPerSec: number
): number {
  if (hourlyPrice <= 0 || tokensPerSec <= 0) return 0;
  const tokensPerHour = tokensPerSec * 3600;
  const tokensPerDollar = tokensPerHour / hourlyPrice;
  return Number((tokensPerDollar / 1e6).toFixed(2)); // Return in Millions of Tokens
}

/**
 * Buy vs Rent TCO Break-Even Calculator
 */
export interface TcoResult {
  hardwareUpfrontCost: number;
  monthlyElectricityCost: number;
  monthlyCloudCost: number;
  breakEvenMonths: number;
  savings1Year: number;
  savings2Year: number;
}

export function calculateTcoBreakEven(
  gpuCount: number,
  gpuMsrp: number,
  rigBaseCost: number = 1200,
  powerWattsPerGpu: number = 450,
  kwhRate: number = 0.14,
  cloudHourlyPrice: number = 0.79
): TcoResult {
  const totalHardwareCost = gpuCount * gpuMsrp + rigBaseCost;
  // Total power in kW (including ~150W for system base):
  const totalPowerKw = (gpuCount * powerWattsPerGpu + 150) / 1000;
  // Monthly hours = 730
  const monthlyKwh = totalPowerKw * 730;
  const monthlyElectricityCost = monthlyKwh * kwhRate;

  const monthlyCloudCost = gpuCount * cloudHourlyPrice * 730;
  const monthlyNetSavings = monthlyCloudCost - monthlyElectricityCost;

  const breakEvenMonths =
    monthlyNetSavings > 0 ? Number((totalHardwareCost / monthlyNetSavings).toFixed(1)) : 999;

  const savings1Year = Number((monthlyNetSavings * 12 - totalHardwareCost).toFixed(0));
  const savings2Year = Number((monthlyNetSavings * 24 - totalHardwareCost).toFixed(0));

  return {
    hardwareUpfrontCost: totalHardwareCost,
    monthlyElectricityCost: Number(monthlyElectricityCost.toFixed(2)),
    monthlyCloudCost: Number(monthlyCloudCost.toFixed(2)),
    breakEvenMonths,
    savings1Year,
    savings2Year,
  };
}
