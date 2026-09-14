import React, { useState, useMemo, useEffect } from 'react';
import { VramCalculator } from './VramCalculator';
import { PricingTable } from './PricingTable';
import { FaqSection } from './FaqSection';
import {
  PRESET_MODELS,
  calculateVramRequirements,
  type VramConfig,
  type ModelPreset,
  type QuantizationType,
  type RunMode,
} from '../../lib/calculations';
import {
  Copy,
  Check,
  Terminal,
  Share2,
} from 'lucide-react';

interface GpuCalcAppProps {
  initialModelId?: string;
  hideFaq?: boolean;
}

export const GpuCalcApp: React.FC<GpuCalcAppProps> = ({
  initialModelId,
  hideFaq = false,
}) => {
  // Find initial preset or fallback to Llama 3.3 70B
  const defaultPreset = useMemo(() => {
    if (initialModelId) {
      const found = PRESET_MODELS.find((p) => p.id === initialModelId || p.slug === initialModelId);
      if (found) return found;
    }
    return PRESET_MODELS[0];
  }, [initialModelId]);

  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(defaultPreset.id);

  const [config, setConfig] = useState<VramConfig>({
    parametersB: defaultPreset.parametersB,
    quantization: defaultPreset.parametersB >= 70 ? 'int4' : 'fp16',
    contextLength: defaultPreset.defaultContext || 8192,
    batchSize: 1,
    mode: 'inference',
    kvCacheQuantization: 'fp16',
    tensorParallelism: 1,
    isVlm: defaultPreset.isVlm || false,
    imageCount: defaultPreset.isVlm ? 1 : 0,
    imageResolution: 1024,
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

  // Restore state from URL query parameters on mount (Client-side)
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const params = new URLSearchParams(window.location.search);
      const urlModel = params.get('model');
      const urlQuant = params.get('quant') as QuantizationType | null;
      const urlCtx = params.get('ctx');
      const urlBatch = params.get('batch');
      const urlMode = params.get('mode') as RunMode | null;
      const urlTp = params.get('tp');

      if (urlModel) {
        const match = PRESET_MODELS.find((p) => p.id === urlModel || p.slug === urlModel);
        if (match) {
          setSelectedPresetId(match.id);
          setConfig((prev) => ({
            ...prev,
            parametersB: match.parametersB,
            contextLength: urlCtx ? parseInt(urlCtx) : (match.defaultContext || 8192),
            quantization: urlQuant || (match.parametersB >= 70 ? 'int4' : 'fp16'),
            batchSize: urlBatch ? parseInt(urlBatch) : 1,
            mode: urlMode || 'inference',
            tensorParallelism: urlTp ? parseInt(urlTp) : 1,
            isVlm: match.isVlm || false,
            imageCount: match.isVlm ? 1 : 0,
            layers: match.layers,
            heads: match.heads,
            kvHeads: match.kvHeads,
            headDim: match.headDim,
            hiddenDim: match.hiddenDim,
          }));
          return;
        }
      }

      if (urlQuant || urlCtx || urlBatch || urlMode || urlTp) {
        setConfig((prev) => ({
          ...prev,
          quantization: urlQuant || prev.quantization,
          contextLength: urlCtx ? parseInt(urlCtx) : prev.contextLength,
          batchSize: urlBatch ? parseInt(urlBatch) : prev.batchSize,
          mode: urlMode || prev.mode,
          tensorParallelism: urlTp ? parseInt(urlTp) : prev.tensorParallelism,
        }));
      }
    } catch (e) {
      // URL parsing fallback
    }
  }, []);

  // Synchronize state to URL without reload
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const params = new URLSearchParams();
      if (selectedPresetId) params.set('model', selectedPresetId);
      params.set('quant', config.quantization);
      params.set('ctx', config.contextLength.toString());
      if (config.batchSize > 1) params.set('batch', config.batchSize.toString());
      if (config.mode !== 'inference') params.set('mode', config.mode);
      if (config.tensorParallelism && config.tensorParallelism > 1) params.set('tp', config.tensorParallelism.toString());

      const newUrl = `${window.location.pathname}?${params.toString()}`;
      window.history.replaceState({}, '', newUrl);
    } catch (e) {
      // replaceState fallback
    }
  }, [config, selectedPresetId]);

  const [copied, setCopied] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);
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
GPU Cluster: ${config.tensorParallelism || 1}x GPUs
-------------------------
Base Weights: ${breakdown.modelWeightsGb} GB
KV-Cache: ${breakdown.kvCacheGb} GB
Total Required VRAM: ${breakdown.totalVramGb} GB (VRAM / GPU: ${breakdown.vramPerGpuGb} GB)
Recommended Allocation: ${breakdown.recommendedVramGb} GB (with 15% safety buffer)`;

    navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleShareLink = () => {
    if (typeof window !== 'undefined') {
      navigator.clipboard.writeText(window.location.href);
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 2000);
    }
  };

  // Generate vLLM / Ollama command snippet
  const cliSnippet = useMemo(() => {
    const modelName = selectedPresetId
      ? PRESET_MODELS.find((p) => p.id === selectedPresetId)?.name || 'custom-model'
      : 'custom-model';
    if (config.mode === 'inference') {
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
      const tpFlag =
        config.tensorParallelism && config.tensorParallelism > 1
          ? `--tensor-parallel-size ${config.tensorParallelism}`
          : '';
      return `vllm serve "${modelName}" \\
  --max-model-len ${config.contextLength} \\
  --gpu-memory-utilization 0.90 ${quantFlag} ${kvFlag} ${tpFlag}`;
    } else {
      const numGpus = config.tensorParallelism || 1;
      return `accelerate launch --num_processes=${numGpus} train.py \\
  --model_name_or_path "${modelName}" \\
  --max_seq_length ${config.contextLength} \\
  --per_device_train_batch_size ${config.batchSize} \\
  --gradient_checkpointing True \\
  --use_qlora True`;
    }
  }, [config, selectedPresetId]);

  return (
    <div className="space-y-8">
      {/* Top Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-neutral-900/40 border border-neutral-800">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-neutral-400" />
          <span className="text-xs font-medium text-neutral-300">
            Real-time VRAM Sizing & Cloud Matrix
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleShareLink}
            className="px-2.5 py-1.5 rounded-lg bg-neutral-900 hover:bg-neutral-800 text-neutral-300 hover:text-white border border-neutral-800 text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
            title="Copy shareable permalink with current configuration"
          >
            {copiedLink ? <Check className="w-3.5 h-3.5 text-neutral-200" /> : <Share2 className="w-3.5 h-3.5 text-neutral-400" />}
            <span>{copiedLink ? 'Link Copied' : 'Share'}</span>
          </button>

          <button
            onClick={() => setShowCliSnippet(!showCliSnippet)}
            className="px-2.5 py-1.5 rounded-lg bg-neutral-900 hover:bg-neutral-800 text-neutral-300 hover:text-white border border-neutral-800 text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Terminal className="w-3.5 h-3.5 text-neutral-400" />
            <span>{showCliSnippet ? 'Hide CLI' : 'Deploy CLI'}</span>
          </button>

          <button
            onClick={handleCopySummary}
            className="px-2.5 py-1.5 rounded-lg bg-white text-neutral-950 hover:bg-neutral-200 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer dark:bg-white dark:text-neutral-950"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-neutral-900" /> : <Copy className="w-3.5 h-3.5 text-neutral-900" />}
            <span>{copied ? 'Copied!' : 'Copy Summary'}</span>
          </button>
        </div>
      </div>

      {/* CLI Launch Snippet (Collapsible) */}
      {showCliSnippet && (
        <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3.5 font-mono text-xs text-neutral-300 space-y-2">
          <div className="flex items-center justify-between text-neutral-400 text-[11px]">
            <span>Generated Launch Command (vLLM / Accelerate):</span>
            <button
              onClick={() => {
                navigator.clipboard.writeText(cliSnippet);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className="hover:text-white flex items-center gap-1 cursor-pointer"
            >
              <Copy className="w-3 h-3" /> Copy
            </button>
          </div>
          <pre className="p-3 rounded-lg bg-neutral-900/60 border border-neutral-800 overflow-x-auto text-neutral-200">
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

      {/* Section 3: Technical FAQ Accordion */}
      {!hideFaq && (
        <div id="faq" className="pt-6">
          <FaqSection />
        </div>
      )}
    </div>
  );
};

export default GpuCalcApp;
