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
      <div className="rounded-2xl border border-slate-800/80 bg-slate-900/60 p-5 backdrop-blur-xl shadow-xl shadow-black/20">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-semibold text-slate-100">Popular Model Presets</h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
              Auto-Architected
            </span>
          </div>
          {selectedPresetId && (
            <button
              onClick={() => onSelectPreset(null)}
              className="text-xs flex items-center gap-1 text-slate-400 hover:text-slate-200 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset to Custom
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {PRESET_MODELS.map((p) => {
            const isSelected = selectedPresetId === p.id;
            return (
              <button
                key={p.id}
                onClick={() => handlePresetClick(p)}
                className={`group relative text-xs font-medium px-3.5 py-2 rounded-xl transition-all duration-200 border ${
                  isSelected
                    ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white border-indigo-400 shadow-lg shadow-indigo-500/25 ring-2 ring-indigo-400/40'
                    : 'bg-slate-800/60 hover:bg-slate-800 text-slate-300 hover:text-white border-slate-700/60 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <span>{p.name}</span>
                  {p.isMoe && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 font-mono">
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
      <div className="rounded-2xl border border-slate-800/80 bg-slate-900/60 p-6 backdrop-blur-xl shadow-xl shadow-black/20 space-y-6">
        {/* Mode Selector Tabs */}
        <div>
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2.5">
            Operational Workload & Target Mode
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
                  className={`p-3 rounded-xl border text-left transition-all duration-200 ${
                    isSelected
                      ? 'bg-indigo-950/60 border-indigo-500/80 text-white shadow-lg shadow-indigo-900/30 ring-1 ring-indigo-500/50'
                      : 'bg-slate-800/40 border-slate-800 hover:border-slate-700 text-slate-300 hover:bg-slate-800/80'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    <Icon className={`w-4 h-4 ${isSelected ? 'text-indigo-400' : 'text-slate-400'}`} />
                    <span className="font-semibold text-xs text-slate-100">{m.label}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-1">{m.desc}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Sliders and Selectors Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 border-t border-slate-800/80">
          {/* Parameter Count */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-indigo-400" />
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
                  className="w-20 px-2 py-1 text-right text-xs font-mono font-bold bg-slate-950 border border-slate-700 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500"
                />
                <span className="text-xs text-slate-400 font-mono">B</span>
              </div>
            </div>

            <input
              type="range"
              min="0.5"
              max="140"
              step="0.5"
              value={config.parametersB <= 140 ? config.parametersB : 140}
              onChange={(e) => handleCustomParamChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
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
              <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                Weight Precision / Quantization
              </label>
              <span className="text-xs font-mono text-cyan-300 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
                {QUANT_BYTES[config.quantization].bytes} B/param
              </span>
            </div>

            <select
              value={config.quantization}
              onChange={(e) => updateConfig({ quantization: e.target.value as QuantizationType })}
              className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-xs font-medium text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
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
              <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-violet-400" />
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
                  className="w-24 px-2 py-1 text-right text-xs font-mono font-bold bg-slate-950 border border-slate-700 rounded-lg text-violet-300 focus:outline-none focus:border-violet-500"
                />
                <span className="text-xs text-slate-400 font-mono">tokens</span>
              </div>
            </div>

            <input
              type="range"
              min="512"
              max="131072"
              step="512"
              value={config.contextLength}
              onChange={(e) => updateConfig({ contextLength: parseInt(e.target.value) })}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
            />

            <div className="flex flex-wrap gap-1.5 pt-1">
              {contextPresets.map((c) => (
                <button
                  key={c}
                  onClick={() => updateConfig({ contextLength: c })}
                  className={`text-[10px] font-mono px-2 py-0.5 rounded-md transition-colors border ${
                    config.contextLength === c
                      ? 'bg-violet-900/60 text-violet-200 border-violet-500/60'
                      : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                  }`}
                >
                  {c >= 1024 ? `${c / 1024}k` : c}
                </button>
              ))}
            </div>
          </div>

          {/* Batch Size & KV Quantization */}
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-200 block mb-1.5">
                  Batch Size / Concurrency
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    max="128"
                    value={config.batchSize}
                    onChange={(e) => updateConfig({ batchSize: Math.max(1, parseInt(e.target.value) || 1) })}
                    className="w-full px-3 py-1.5 bg-slate-950 border border-slate-700 rounded-xl text-xs font-mono font-bold text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                  <span className="text-xs text-slate-400 font-mono">reqs</span>
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-200 block mb-1.5">
                  KV-Cache Precision
                </label>
                <select
                  value={config.kvCacheQuantization}
                  onChange={(e) =>
                    updateConfig({ kvCacheQuantization: e.target.value as KvCacheQuantization })
                  }
                  className="w-full px-2.5 py-1.5 bg-slate-950 border border-slate-700 rounded-xl text-xs font-medium text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                >
                  <option value="fp16">FP16 (2.0 B)</option>
                  <option value="fp8">FP8 (1.0 B - vLLM)</option>
                  <option value="int4">INT4 (0.5 B - FlashInfer)</option>
                </select>
              </div>
            </div>

            {config.mode.includes('lora') && (
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs flex items-center justify-between">
                <span className="text-slate-300">Trainable LoRA Adapters:</span>
                <span className="font-mono text-indigo-400 font-bold">
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
        <div className="md:col-span-1 rounded-2xl border border-indigo-500/40 bg-gradient-to-br from-indigo-950/80 via-slate-900/90 to-slate-950 p-6 backdrop-blur-xl shadow-xl shadow-indigo-950/40 flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none" />

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">
                Minimum Required VRAM
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Active
              </span>
            </div>

            <div className="flex items-baseline gap-2 my-2">
              <span className="text-5xl font-extrabold tracking-tight text-white font-mono">
                {breakdown.totalVramGb}
              </span>
              <span className="text-lg font-bold text-indigo-300 font-mono">GB</span>
            </div>

            <div className="flex items-center gap-1.5 text-xs text-slate-300 mt-1">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Recommended Target:</span>
              <span className="font-mono font-bold text-emerald-300">
                {breakdown.recommendedVramGb} GB
              </span>
              <span className="text-[10px] text-slate-500">(+15% buffer)</span>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-indigo-500/20 space-y-2">
            <span className="text-[11px] font-semibold text-slate-300 block">
              Suggested Minimum GPU Tier:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {breakdown.recommendedVramGb <= 24 ? (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  <Check className="w-3.5 h-3.5" /> 1x RTX 3090 / 4090 (24GB)
                </span>
              ) : breakdown.recommendedVramGb <= 32 ? (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  <Check className="w-3.5 h-3.5" /> 1x RTX 5090 (32GB)
                </span>
              ) : breakdown.recommendedVramGb <= 48 ? (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  <Check className="w-3.5 h-3.5" /> 1x L40S / RTX 6000 Ada (48GB)
                </span>
              ) : breakdown.recommendedVramGb <= 80 ? (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                  <Check className="w-3.5 h-3.5" /> 1x A100 / H100 (80GB)
                </span>
              ) : breakdown.recommendedVramGb <= 160 ? (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-purple-500/15 text-purple-300 border border-purple-500/30">
                  <Check className="w-3.5 h-3.5" /> 2x A100 / H100 (80GB)
                </span>
              ) : breakdown.recommendedVramGb <= 320 ? (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30">
                  <Check className="w-3.5 h-3.5" /> 4x A100 / H100 (80GB)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-rose-500/15 text-rose-300 border border-rose-500/30">
                  <AlertTriangle className="w-3.5 h-3.5" /> 8x H100 / B200 Supercluster
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Memory Distribution Visual Breakdown */}
        <div className="md:col-span-2 rounded-2xl border border-slate-800/80 bg-slate-900/60 p-6 backdrop-blur-xl shadow-xl shadow-black/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-indigo-400" />
                VRAM Memory Footprint Breakdown
              </h3>
              <span className="text-xs font-mono text-slate-400">
                Sum: {breakdown.totalVramGb} GB
              </span>
            </div>

            {/* Stacked Progress Bar */}
            <div className="h-3.5 w-full bg-slate-950 rounded-full overflow-hidden flex border border-slate-800 mb-4 p-0.5">
              {breakdown.modelWeightsGb > 0 && (
                <div
                  style={{ width: `${(breakdown.modelWeightsGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-indigo-500 rounded-l-full h-full transition-all duration-300"
                  title={`Weights: ${breakdown.modelWeightsGb} GB`}
                />
              )}
              {breakdown.kvCacheGb > 0 && (
                <div
                  style={{ width: `${(breakdown.kvCacheGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-cyan-400 h-full transition-all duration-300"
                  title={`KV-Cache: ${breakdown.kvCacheGb} GB`}
                />
              )}
              {breakdown.optimizerGb > 0 && (
                <div
                  style={{ width: `${(breakdown.optimizerGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-amber-400 h-full transition-all duration-300"
                  title={`Optimizer: ${breakdown.optimizerGb} GB`}
                />
              )}
              {breakdown.gradientsGb > 0 && (
                <div
                  style={{ width: `${(breakdown.gradientsGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-rose-400 h-full transition-all duration-300"
                  title={`Gradients: ${breakdown.gradientsGb} GB`}
                />
              )}
              {breakdown.activationsGb > 0 && (
                <div
                  style={{ width: `${(breakdown.activationsGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-purple-400 h-full transition-all duration-300"
                  title={`Activations: ${breakdown.activationsGb} GB`}
                />
              )}
              {breakdown.cudaOverheadGb > 0 && (
                <div
                  style={{ width: `${(breakdown.cudaOverheadGb / breakdown.totalVramGb) * 100}%` }}
                  className="bg-slate-600 rounded-r-full h-full transition-all duration-300"
                  title={`CUDA Context: ${breakdown.cudaOverheadGb} GB`}
                />
              )}
            </div>

            {/* Breakdown item cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-indigo-500 shrink-0" />
                  <span>Model Weights</span>
                </div>
                <div className="mt-1 text-sm font-bold font-mono text-slate-100">
                  {breakdown.modelWeightsGb} <span className="text-xs text-slate-400 font-normal">GB</span>
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 shrink-0" />
                  <span>KV-Cache Context</span>
                </div>
                <div className="mt-1 text-sm font-bold font-mono text-slate-100">
                  {breakdown.kvCacheGb} <span className="text-xs text-slate-400 font-normal">GB</span>
                </div>
              </div>

              {config.mode !== 'inference' ? (
                <>
                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                      <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" />
                      <span>Optimizer States</span>
                    </div>
                    <div className="mt-1 text-sm font-bold font-mono text-slate-100">
                      {breakdown.optimizerGb} <span className="text-xs text-slate-400 font-normal">GB</span>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                      <span className="w-2 h-2 rounded-full bg-rose-400 shrink-0" />
                      <span>Gradients</span>
                    </div>
                    <div className="mt-1 text-sm font-bold font-mono text-slate-100">
                      {breakdown.gradientsGb} <span className="text-xs text-slate-400 font-normal">GB</span>
                    </div>
                  </div>
                </>
              ) : null}

              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-purple-400 shrink-0" />
                  <span>Activations</span>
                </div>
                <div className="mt-1 text-sm font-bold font-mono text-slate-100">
                  {breakdown.activationsGb} <span className="text-xs text-slate-400 font-normal">GB</span>
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-slate-500 shrink-0" />
                  <span>CUDA & Runtime</span>
                </div>
                <div className="mt-1 text-sm font-bold font-mono text-slate-100">
                  {breakdown.cudaOverheadGb} <span className="text-xs text-slate-400 font-normal">GB</span>
                </div>
              </div>
            </div>
          </div>

          {/* Performance throughput estimate preview */}
          {config.mode === 'inference' && breakdown.totalVramGb <= 80 && (
            <div className="mt-4 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 text-slate-300">
                <Gauge className="w-4 h-4 text-emerald-400" />
                <span>Estimated Decoding Speed (Tokens/sec):</span>
              </div>
              <div className="flex items-center gap-3 font-mono text-xs">
                {breakdown.totalVramGb <= 24 && (
                  <span className="text-slate-300">
                    RTX 4090: <b className="text-emerald-400">{rtx4090Speed} t/s</b>
                  </span>
                )}
                {breakdown.totalVramGb <= 80 && (
                  <>
                    <span className="text-slate-300">
                      A100 (80GB): <b className="text-cyan-400">{a100Speed} t/s</b>
                    </span>
                    <span className="text-slate-300">
                      H100 (80GB): <b className="text-indigo-400">{h100Speed} t/s</b>
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
