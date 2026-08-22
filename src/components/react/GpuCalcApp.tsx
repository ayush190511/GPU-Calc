import React, { useState, useMemo } from 'react';
import { VramCalculator } from './VramCalculator';
import { PricingTable } from './PricingTable';
import {
  PRESET_MODELS,
  calculateVramRequirements,
  type VramConfig,
  type ModelPreset,
} from '../../lib/calculations';
import {
  Sparkles,
  Share2,
  Copy,
  Check,
  Terminal,
  Server,
  Layers,
  Zap,
  Bookmark,
} from 'lucide-react';

export const GpuCalcApp: React.FC = () => {
  // Default to Llama 3.3 70B
  const defaultPreset = PRESET_MODELS[0];

  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(defaultPreset.id);

  const [config, setConfig] = useState<VramConfig>({
    parametersB: defaultPreset.parametersB,
    quantization: 'int4',
    contextLength: 8192,
    batchSize: 1,
    mode: 'inference',
    kvCacheQuantization: 'fp16',
    loraRank: 16,
    loraTrainableFraction: 0.012,
    gradientCheckpointing: true,
    cudaOverheadBufferGb: 1.5,
    safetyMarginPct: 0.15,
    layers: defaultPreset.layers,
    heads: defaultPreset.heads,
    kvHeads: defaultPreset.kvHeads,
    headDim: defaultPreset.headDim,
    hiddenDim: defaultPreset.hiddenDim,
  });

  const [copied, setCopied] = useState(false);
  const [showCliSnippet, setShowCliSnippet] = useState(false);

  // Compute breakdown dynamically
  const breakdown = useMemo(() => {
    return calculateVramRequirements(config);
  }, [config]);

  const handleSelectPreset = (preset: ModelPreset | null) => {
    setSelectedPresetId(preset ? preset.id : null);
  };

  const handleCopySummary = () => {
    const summary = `GPUCalc Estimation:
Model: ${selectedPresetId ? PRESET_MODELS.find((p) => p.id === selectedPresetId)?.name : `${config.parametersB}B Custom`}
Mode: ${config.mode}
Precision: ${config.quantization}
Context Window: ${config.contextLength.toLocaleString()} tokens
Batch Size: ${config.batchSize}
-------------------------
Base Weights: ${breakdown.modelWeightsGb} GB
KV-Cache: ${breakdown.kvCacheGb} GB
Total Required VRAM: ${breakdown.totalVramGb} GB
Recommended Allocation: ${breakdown.recommendedVramGb} GB (with 15% safety buffer)`;

    navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Generate vLLM / Ollama command snippet
  const cliSnippet = useMemo(() => {
    if (config.mode === 'inference') {
      const isQuant = config.quantization === 'int4' || config.quantization === 'fp8';
      const quantFlag =
        config.quantization === 'int4'
          ? '--quantization awq'
          : config.quantization === 'fp8'
            ? '--quantization fp8'
            : '';
      const kvFlag =
        config.kvCacheQuantization === 'fp8'
          ? '--kv-cache-dtype fp8'
          : '';
      return `vllm serve meta-llama/Llama-3.3-70B-Instruct \\
  --max-model-len ${config.contextLength} \\
  --gpu-memory-utilization 0.90 ${quantFlag} ${kvFlag}`;
    } else {
      return `accelerate launch train.py \\
  --model_name_or_path meta-llama/Llama-3.3-70B-Instruct \\
  --max_seq_length ${config.contextLength} \\
  --per_device_train_batch_size ${config.batchSize} \\
  --gradient_checkpointing True \\
  --use_qlora True`;
    }
  }, [config]);

  return (
    <div className="space-y-10">
      {/* Top Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-slate-900/40 border border-slate-800/80 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-semibold text-slate-300">
            Interactive VRAM & Pricing Engine Active
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowCliSnippet(!showCliSnippet)}
            className="px-3 py-1.5 rounded-xl bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/60 text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>{showCliSnippet ? 'Hide CLI' : 'Deploy Command'}</span>
          </button>

          <button
            onClick={handleCopySummary}
            className="px-3 py-1.5 rounded-xl bg-indigo-600/80 hover:bg-indigo-600 text-white text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm shadow-indigo-600/20"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-300" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied Specs!' : 'Copy Summary'}</span>
          </button>
        </div>
      </div>

      {/* CLI Launch Snippet (Collapsible) */}
      {showCliSnippet && (
        <div className="rounded-2xl border border-cyan-500/30 bg-slate-950/90 p-4 font-mono text-xs text-cyan-300 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-[11px]">
            <span>Generated Launch Command (vLLM / HuggingFace Accelerate):</span>
            <button
              onClick={() => {
                navigator.clipboard.writeText(cliSnippet);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className="hover:text-white flex items-center gap-1"
            >
              <Copy className="w-3 h-3" /> Copy
            </button>
          </div>
          <pre className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 overflow-x-auto text-slate-100">
            {cliSnippet}
          </pre>
        </div>
      )}

      {/* Section 1: VRAM Sizing Calculator */}
      <VramCalculator
        config={config}
        onChange={setConfig}
        breakdown={breakdown}
        selectedPresetId={selectedPresetId}
        onSelectPreset={handleSelectPreset}
      />

      {/* Section 2: Real-time Cloud GPU Pricing Matrix */}
      <div id="pricing-matrix" className="pt-2">
        <PricingTable requiredVramGb={breakdown.totalVramGb} />
      </div>
    </div>
  );
};

export default GpuCalcApp;
