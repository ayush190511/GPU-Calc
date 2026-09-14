import React from 'react';
import {
  PRESET_MODELS,
  QUANT_BYTES,
  estimateInferenceThroughput,
  type VramConfig,
  type VramBreakdown,
  type QuantizationType,
  type RunMode,
  type KvCacheQuantization,
  type ModelPreset,
} from '../../lib/calculations';
import {
  Cpu,
  Layers,
  Sparkles,
  Zap,
  Sliders,
  HardDrive,
  ShieldCheck,
  Activity,
  AlertTriangle,
  RotateCcw,
  Gauge,
  Check,
} from 'lucide-react';

interface VramCalculatorProps {
  config: VramConfig;
  onChange: (config: VramConfig) => void;
  breakdown: VramBreakdown;
  selectedPresetId: string | null;
  onSelectPreset: (preset: ModelPreset | null) => void;
}

export const VramCalculator: React.FC<VramCalculatorProps> = ({
  config,
  onChange,
  breakdown,
  selectedPresetId,
  onSelectPreset,
}) => {
  const updateConfig = (partial: Partial<VramConfig>) => {
    onChange({ ...config, ...partial });
  };

  const handlePresetClick = (preset: ModelPreset) => {
    onSelectPreset(preset);
    onChange({
      ...config,
      parametersB: preset.parametersB,
      contextLength: Math.min(config.contextLength, preset.maxContext),
      layers: preset.layers,
      heads: preset.heads,
      kvHeads: preset.kvHeads,
      headDim: preset.headDim,
      hiddenDim: preset.hiddenDim,
    });
  };

  const handleCustomParamChange = (val: number) => {
    onSelectPreset(null);
    updateConfig({
      parametersB: val,
      layers: undefined,
      heads: undefined,
      kvHeads: undefined,
      headDim: undefined,
      hiddenDim: undefined,
    });
  };

  // Quick context presets
  const contextPresets = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

  // Throughput estimate for RTX 4090 / A100 / H100
  const rtx4090Speed = estimateInferenceThroughput(breakdown.modelWeightsGb, breakdown.kvCacheGb, 1008);
  const a100Speed = estimateInferenceThroughput(breakdown.modelWeightsGb, breakdown.kvCacheGb, 2039);
  const h100Speed = estimateInferenceThroughput(breakdown.modelWeightsGb, breakdown.kvCacheGb, 3350);

  return (
    <div className="space-y-6">
      {/* Model Presets Selector */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/30 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-neutral-400" />
            <h2 className="text-sm font-semibold text-neutral-200">Model Presets</h2>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 font-mono">
              Auto-Configured
            </span>
          </div>
          {selectedPresetId && (
            <button
              onClick={() => onSelectPreset(null)}
              className="text-xs flex items-center gap-1 text-neutral-400 hover:text-white transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset to Custom
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {PRESET_MODELS.map((p) => {
            const isSelected = selectedPresetId === p.id;
            return (
              <button
                key={p.id}
                onClick={() => handlePresetClick(p)}
                className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-all duration-150 border cursor-pointer ${
                  isSelected
                    ? 'bg-white text-neutral-950 font-semibold border-white shadow-sm dark:bg-white dark:text-neutral-950'
                    : 'bg-neutral-900/60 hover:bg-neutral-800 text-neutral-300 hover:text-white border-neutral-800 hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <span>{p.name}</span>
                  {p.isMoe && (
                    <span className={`text-[9px] px-1 rounded font-mono ${isSelected ? 'bg-neutral-200 text-neutral-800' : 'bg-neutral-800 text-neutral-400'}`}>
                      MoE
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Configuration Card */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/30 p-5 space-y-5">
        {/* Mode Selector Tabs */}
        <div>
          <label className="block text-xs font-semibold text-neutral-300 uppercase tracking-wider mb-2">
            Target Workload
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            {[
              { id: 'inference', label: 'Inference', desc: 'Serving & Generation', icon: Zap },
              { id: 'qlora', label: 'QLoRA 4-bit', desc: 'Fine-Tune Adapters', icon: Sliders },
              { id: 'lora_16bit', label: 'LoRA 16-bit', desc: 'Standard Adapters', icon: Layers },
              { id: 'full_finetune_16bit', label: 'Full Tune 16-bit', desc: 'Mixed Precision AdamW', icon: Activity },
              { id: 'full_finetune_32bit', label: 'Full Tune 32-bit', desc: 'Pure FP32 Training', icon: Cpu },
            ].map((m) => {
              const Icon = m.icon;
              const isSelected = config.mode === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => updateConfig({ mode: m.id as RunMode })}
                  className={`p-2.5 rounded-lg border text-left transition-all duration-150 cursor-pointer ${
                    isSelected
                      ? 'bg-neutral-800 border-neutral-600 text-white shadow-sm'
                      : 'bg-neutral-950/40 border-neutral-800/80 hover:border-neutral-700 text-neutral-400 hover:text-neutral-200'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    <Icon className={`w-3.5 h-3.5 ${isSelected ? 'text-white' : 'text-neutral-400'}`} />
                    <span className="font-semibold text-xs text-neutral-100">{m.label}</span>
                  </div>
                  <p className="text-[11px] text-neutral-400 line-clamp-1">{m.desc}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Sliders and Selectors Grid */}
        {/* Sliders and Selectors Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 border-t border-neutral-800">
          {/* Parameter Count */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-neutral-200 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-neutral-400" />
                Model Parameters (Billion)
              </label>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  min="0.1"
                  max="1000"
                  step="0.1"
                  value={config.parametersB}
                  onChange={(e) => handleCustomParamChange(parseFloat(e.target.value) || 1)}
                  className="w-20 px-2 py-1 text-right text-xs font-mono font-bold bg-neutral-950 border border-neutral-800 rounded-md text-white focus:outline-none focus:border-neutral-500"
                />
                <span className="text-xs text-neutral-400 font-mono">B</span>
              </div>
            </div>

            <input
              type="range"
              min="0.5"
              max="140"
              step="0.5"
              value={config.parametersB <= 140 ? config.parametersB : 140}
              onChange={(e) => handleCustomParamChange(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-white"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
              <span>0.5B (Edge)</span>
              <span>7B</span>
              <span>14B</span>
              <span>32B</span>
              <span>70B</span>
              <span>140B+</span>
            </div>
          </div>

          {/* Model Weights Quantization / Precision */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-neutral-200 flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5 text-neutral-400" />
                Weight Precision / Quantization
              </label>
              <span className="text-xs font-mono text-neutral-300 bg-neutral-800 px-2 py-0.5 rounded border border-neutral-700">
                {QUANT_BYTES[config.quantization].bytes} B/param
              </span>
            </div>

            <select
              value={config.quantization}
              onChange={(e) => updateConfig({ quantization: e.target.value as QuantizationType })}
              className="w-full px-3 py-2 bg-neutral-950 border border-neutral-800 rounded-lg text-xs font-medium text-neutral-200 focus:outline-none focus:border-neutral-500 cursor-pointer"
            >
              <option value="bf16">BF16 / FP16 (2.0 Bytes - Native Master Quality)</option>
              <option value="fp8">FP8 (1.0 Byte - Ada / Hopper Native)</option>
              <option value="int8">INT8 / bitsandbytes (1.0 Byte)</option>
              <option value="int4">INT4 / AWQ / GPTQ (~0.55 Bytes)</option>
              <option value="gguf_q4_k_m">GGUF Q4_K_M (~0.58 Bytes - llama.cpp)</option>
              <option value="gguf_q5_k_m">GGUF Q5_K_M (~0.70 Bytes)</option>
              <option value="gguf_q8_0">GGUF Q8_0 (~1.05 Bytes)</option>
              <option value="fp32">FP32 (4.0 Bytes - Full Single Precision)</option>
            </select>
          </div>

          {/* Context Length */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-neutral-200 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-neutral-400" />
                Context Window (Tokens)
              </label>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  min="512"
                  max="262144"
                  step="512"
                  value={config.contextLength}
                  onChange={(e) => updateConfig({ contextLength: parseInt(e.target.value) || 2048 })}
                  className="w-24 px-2 py-1 text-right text-xs font-mono font-bold bg-neutral-950 border border-neutral-800 rounded-md text-white focus:outline-none focus:border-neutral-500"
                />
                <span className="text-xs text-neutral-400 font-mono">tokens</span>
              </div>
            </div>

            <input
              type="range"
              min="512"
              max="131072"
              step="512"
              value={config.contextLength}
              onChange={(e) => updateConfig({ contextLength: parseInt(e.target.value) })}
              className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-white"
            />

            <div className="flex flex-wrap gap-1.5 pt-1">
              {contextPresets.map((c) => (
                <button
                  key={c}
                  onClick={() => updateConfig({ contextLength: c })}
                  className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors border cursor-pointer ${
                    config.contextLength === c
                      ? 'bg-white text-neutral-950 font-bold border-white dark:bg-white dark:text-neutral-950'
                      : 'bg-neutral-950 text-neutral-400 border-neutral-800 hover:text-white hover:border-neutral-700'
                  }`}
                >
                  {c >= 1024 ? `${c / 1024}k` : c}
                </button>
              ))}
            </div>
          </div>

          {/* Batch Size & KV Quantization & Tensor Parallelism */}
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Batch Size */}
              <div>
                <label className="text-xs font-semibold text-neutral-200 block mb-1.5">
                  Batch Size
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    max="128"
                    value={config.batchSize}
                    onChange={(e) => updateConfig({ batchSize: Math.max(1, parseInt(e.target.value) || 1) })}
                    className="w-full px-2.5 py-1.5 bg-neutral-950 border border-neutral-800 rounded-lg text-xs font-mono font-bold text-white focus:outline-none focus:border-neutral-500"
                  />
                  <span className="text-xs text-neutral-400 font-mono">reqs</span>
                </div>
              </div>

              {/* KV Precision */}
              <div>
                <label className="text-xs font-semibold text-neutral-200 block mb-1.5">
                  KV Precision
                </label>
                <select
                  value={config.kvCacheQuantization}
                  onChange={(e) =>
                    updateConfig({ kvCacheQuantization: e.target.value as KvCacheQuantization })
                  }
                  className="w-full px-2 py-1.5 bg-neutral-950 border border-neutral-800 rounded-lg text-xs font-medium text-neutral-200 focus:outline-none focus:border-neutral-500 cursor-pointer"
                >
                  <option value="fp16">FP16 (2.0 B)</option>
                  <option value="fp8">FP8 (1.0 B)</option>
                  <option value="int4">INT4 (0.5 B)</option>
                </select>
              </div>

              {/* Multi-GPU Tensor Parallelism (TP) */}
              <div>
                <label className="text-xs font-semibold text-neutral-200 block mb-1.5 flex items-center justify-between">
                  <span>GPU Cluster</span>
                  <span className="text-[10px] font-mono text-neutral-300 font-bold">
                    {config.tensorParallelism || 1}x GPU{config.tensorParallelism && config.tensorParallelism > 1 ? 's' : ''}
                  </span>
                </label>
                <div className="grid grid-cols-4 gap-1">
                  {[1, 2, 4, 8].map((tp) => {
                    const isSelected = (config.tensorParallelism || 1) === tp;
                    return (
                      <button
                        key={tp}
                        type="button"
                        onClick={() => updateConfig({ tensorParallelism: tp })}
                        className={`py-1 text-xs font-mono font-bold rounded border transition-all cursor-pointer ${
                          isSelected
                            ? 'bg-white text-neutral-950 border-white font-bold dark:bg-white dark:text-neutral-950'
                            : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700'
                        }`}
                      >
                        {tp}x
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Multimodal / Vision Language Model (VLM) Image Token Expansion */}
            <div className="p-3.5 rounded-xl bg-neutral-950 border border-neutral-800 space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-neutral-200 flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.isVlm || false}
                    onChange={(e) => updateConfig({ isVlm: e.target.checked, imageCount: e.target.checked ? (config.imageCount || 1) : 0 })}
                    className="w-4 h-4 rounded text-white focus:ring-0 accent-neutral-900 bg-neutral-900 border-neutral-700"
                  />
                  <span>Multimodal / Vision-Language Model (VLM) Tokens</span>
                </label>
                {config.isVlm && (
                  <span className="text-[10px] font-mono text-neutral-200 bg-neutral-900 px-2 py-0.5 rounded border border-neutral-700">
                    +{breakdown.vlmImageTokens.toLocaleString()} Image Tokens
                  </span>
                )}
              </div>

              {config.isVlm && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-neutral-800">
                  <div>
                    <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
                      <span>Input Images per Request:</span>
                      <span className="font-mono text-neutral-200 font-bold">{config.imageCount || 1} img</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="8"
                      value={config.imageCount || 1}
                      onChange={(e) => updateConfig({ imageCount: parseInt(e.target.value) || 1 })}
                      className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-white"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
                      <span>Image Resolution:</span>
                      <span className="font-mono text-neutral-200 font-bold">{config.imageResolution || 1024}px</span>
                    </div>
                    <select
                      value={config.imageResolution || 1024}
                      onChange={(e) => updateConfig({ imageResolution: parseInt(e.target.value) })}
                      className="w-full px-2 py-1 bg-neutral-950 border border-neutral-800 rounded-lg text-xs text-neutral-200"
                    >
                      <option value="512">512 × 512 (~400 patch tokens)</option>
                      <option value="1024">1024 × 1024 (~1,600 patch tokens)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Interconnect Requirement Badge if TP > 1 */}
            {breakdown.tensorParallelism > 1 && (
              <div className="p-3 rounded-xl bg-neutral-950 border border-neutral-800 flex items-center justify-between text-xs">
                <span className="text-neutral-300">Inter-GPU Bandwidth Req:</span>
                <span className="font-mono font-bold text-white">
                  {breakdown.interconnectRequirement}
                </span>
              </div>
            )}

            {config.mode.includes('lora') && (
              <div className="p-3 rounded-xl bg-neutral-950 border border-neutral-800 text-xs flex items-center justify-between">
                <span className="text-neutral-300">Trainable LoRA Adapters:</span>
                <span className="font-mono text-white font-bold">
                  {breakdown.trainableParamsB} B params (~1.2%)
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Real-time VRAM Breakdown Hero Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Total Required VRAM Card */}
        <div className="md:col-span-1 rounded-xl border border-neutral-800 bg-neutral-950 p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                Minimum Required VRAM
              </span>
              <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-neutral-800 text-neutral-300 border border-neutral-700">
                Active
              </span>
            </div>

            {breakdown.tensorParallelism > 1 ? (
              <div>
                <div className="flex items-baseline gap-2 my-2">
                  <span className="text-4xl sm:text-5xl font-bold tracking-tight text-white font-mono">
                    {breakdown.vramPerGpuGb}
                  </span>
                  <span className="text-sm font-medium text-neutral-400 font-mono">GB / GPU</span>
                </div>
                <div className="text-[11px] text-neutral-400 font-mono mb-2">
                  Total Cluster: <b className="text-white">{breakdown.totalVramGb} GB</b> across {breakdown.tensorParallelism}x GPUs
                </div>
              </div>
            ) : (
              <div className="flex items-baseline gap-2 my-2">
                <span className="text-5xl font-bold tracking-tight text-white font-mono">
                  {breakdown.totalVramGb}
                </span>
                <span className="text-base font-medium text-neutral-400 font-mono">GB</span>
              </div>
            )}

            <div className="flex items-center gap-1.5 text-xs text-neutral-400 mt-1">
              <ShieldCheck className="w-4 h-4 text-neutral-300 shrink-0" />
              <span>Recommended Target:</span>
              <span className="font-mono font-bold text-white">
                {breakdown.tensorParallelism > 1 ? `${breakdown.recommendedPerGpuVramGb} GB / card` : `${breakdown.recommendedVramGb} GB`}
              </span>
              <span className="text-[10px] text-neutral-500">(+15% buffer)</span>
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-neutral-800/80 space-y-1.5">
            <span className="text-[11px] font-medium text-neutral-400 block">
              Suggested Minimum GPU Tier:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {breakdown.recommendedVramGb <= 24 ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <Check className="w-3 h-3" /> 1x RTX 3090 / 4090 (24GB)
                </span>
              ) : breakdown.recommendedVramGb <= 32 ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <Check className="w-3 h-3" /> 1x RTX 5090 (32GB)
                </span>
              ) : breakdown.recommendedVramGb <= 48 ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <Check className="w-3 h-3" /> 1x L40S / RTX 6000 (48GB)
                </span>
              ) : breakdown.recommendedVramGb <= 80 ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <Check className="w-3 h-3" /> 1x A100 / H100 (80GB)
                </span>
              ) : breakdown.recommendedVramGb <= 160 ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <Check className="w-3 h-3" /> 2x A100 / H100 (80GB)
                </span>
              ) : breakdown.recommendedVramGb <= 320 ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <Check className="w-3 h-3" /> 4x A100 / H100 (80GB)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-800">
                  <AlertTriangle className="w-3 h-3 text-neutral-400" /> 8x H100 / B200 Supercluster
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Memory Distribution Visual Breakdown */}
        <div className="md:col-span-2 rounded-xl border border-neutral-800 bg-neutral-950 p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-300 flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5 text-neutral-400" />
                VRAM Memory Distribution
              </h3>
              <span className="text-xs font-mono text-neutral-400">
                Sum: {breakdown.totalVramGb} GB
              </span>
            </div>

            {/* Stacked Progress Bar */}
            <div className="h-2.5 w-full bg-neutral-900 rounded-full overflow-hidden flex border border-neutral-800 mb-3.5 p-0.5">
              {breakdown.modelWeightsGb > 0 && (
                <div
                  style={{ width: `${(breakdown.modelWeightsGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-neutral-200 rounded-l-full h-full transition-all duration-200"
                  title={`Weights: ${breakdown.modelWeightsGb} GB`}
                />
              )}
              {breakdown.kvCacheGb > 0 && (
                <div
                  style={{ width: `${(breakdown.kvCacheGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-neutral-400 h-full transition-all duration-200"
                  title={`KV-Cache: ${breakdown.kvCacheGb} GB`}
                />
              )}
              {breakdown.optimizerGb > 0 && (
                <div
                  style={{ width: `${(breakdown.optimizerGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-neutral-500 h-full transition-all duration-200"
                  title={`Optimizer: ${breakdown.optimizerGb} GB`}
                />
              )}
              {breakdown.gradientsGb > 0 && (
                <div
                  style={{ width: `${(breakdown.gradientsGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-neutral-600 h-full transition-all duration-200"
                  title={`Gradients: ${breakdown.gradientsGb} GB`}
                />
              )}
              {breakdown.activationsGb > 0 && (
                <div
                  style={{ width: `${(breakdown.activationsGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-neutral-700 h-full transition-all duration-200"
                  title={`Activations: ${breakdown.activationsGb} GB`}
                />
              )}
              {breakdown.cudaOverheadGb > 0 && (
                <div
                  style={{ width: `${(breakdown.cudaOverheadGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-neutral-800 rounded-r-full h-full transition-all duration-200"
                  title={`CUDA Context: ${breakdown.cudaOverheadGb} GB`}
                />
              )}
            </div>

            {/* Breakdown item cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <div className="p-2 rounded-lg bg-neutral-900/50 border border-neutral-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-neutral-200 shrink-0" />
                  <span>Model Weights</span>
                </div>
                <div className="mt-0.5 text-sm font-bold font-mono text-white">
                  {breakdown.modelWeightsGb} <span className="text-xs text-neutral-500 font-normal">GB</span>
                </div>
              </div>

              <div className="p-2 rounded-lg bg-neutral-900/50 border border-neutral-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-neutral-400 shrink-0" />
                  <span>KV-Cache Context</span>
                </div>
                <div className="mt-0.5 text-sm font-bold font-mono text-white">
                  {breakdown.kvCacheGb} <span className="text-xs text-neutral-500 font-normal">GB</span>
                </div>
              </div>

              {config.mode !== 'inference' ? (
                <>
                  <div className="p-2 rounded-lg bg-neutral-900/50 border border-neutral-800/80">
                    <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 shrink-0" />
                      <span>Optimizer States</span>
                    </div>
                    <div className="mt-0.5 text-sm font-bold font-mono text-white">
                      {breakdown.optimizerGb} <span className="text-xs text-neutral-500 font-normal">GB</span>
                    </div>
                  </div>

                  <div className="p-2 rounded-lg bg-neutral-900/50 border border-neutral-800/80">
                    <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-600 shrink-0" />
                      <span>Gradients</span>
                    </div>
                    <div className="mt-0.5 text-sm font-bold font-mono text-white">
                      {breakdown.gradientsGb} <span className="text-xs text-neutral-500 font-normal">GB</span>
                    </div>
                  </div>
                </>
              ) : null}

              <div className="p-2 rounded-lg bg-neutral-900/50 border border-neutral-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-neutral-700 shrink-0" />
                  <span>Activations</span>
                </div>
                <div className="mt-0.5 text-sm font-bold font-mono text-white">
                  {breakdown.activationsGb} <span className="text-xs text-neutral-500 font-normal">GB</span>
                </div>
              </div>

              <div className="p-2 rounded-lg bg-neutral-900/50 border border-neutral-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-neutral-800 shrink-0" />
                  <span>CUDA & Runtime</span>
                </div>
                <div className="mt-0.5 text-sm font-bold font-mono text-white">
                  {breakdown.cudaOverheadGb} <span className="text-xs text-neutral-500 font-normal">GB</span>
                </div>
              </div>
            </div>
          </div>

          {/* Performance throughput estimate preview */}
          {config.mode === 'inference' && breakdown.totalVramGb <= 80 && (
            <div className="mt-3.5 pt-3 border-t border-neutral-800 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-1.5 text-neutral-400">
                <Gauge className="w-3.5 h-3.5 text-neutral-400" />
                <span>Est. Decoding Throughput:</span>
              </div>
              <div className="flex items-center gap-3 font-mono text-xs">
                {breakdown.totalVramGb <= 24 && (
                  <span className="text-neutral-400">
                    RTX 4090: <b className="text-white">{rtx4090Speed} t/s</b>
                  </span>
                )}
                {breakdown.totalVramGb <= 80 && (
                  <>
                    <span className="text-neutral-400">
                      A100: <b className="text-white">{a100Speed} t/s</b>
                    </span>
                    <span className="text-neutral-400">
                      H100: <b className="text-white">{h100Speed} t/s</b>
                    </span>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
