import React, { useState, useMemo } from 'react';
import type { GpuInstance } from '../../lib/types';
import gpuPricingRaw from '../../data/gpu-pricing.json';
import {
  calculateCostPerMillionTokens,
  calculateTokensPerDollar,
  calculateTcoBreakEven,
} from '../../lib/calculations';
import {
  Server,
  ArrowUpDown,
  ExternalLink,
  Filter,
  Search,
  CheckCircle,
  Clock,
  Sparkles,
  TrendingDown,
  Cpu,
  Calculator,
  Zap,
  X,
} from 'lucide-react';

interface PricingTableProps {
  requiredVramGb: number;
}

type SortField =
  | 'spotPricePerHour'
  | 'onDemandPricePerHour'
  | 'totalVramGb'
  | 'memoryBandwidthGbps'
  | 'costPerGb'
  | 'tokensPerDollar';
type SortDirection = 'asc' | 'desc';
type DurationMode = '1h' | '24h' | '100h' | '720h';

export const PricingTable: React.FC<PricingTableProps> = ({ requiredVramGb }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);
  const [selectedArchs, setSelectedArchs] = useState<string[]>([]);
  const [gpuCountFilter, setGpuCountFilter] = useState<'all' | 'single' | 'multi'>('all');
  const [onlyCompatible, setOnlyCompatible] = useState(true);
  const [sortField, setSortField] = useState<SortField>('spotPricePerHour');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [durationMode, setDurationMode] = useState<DurationMode>('1h');
  const [showTcoModal, setShowTcoModal] = useState(false);
  const [tcoGpuModel, setTcoGpuModel] = useState<'rtx4090' | 'rtx5090' | 'a100'>('rtx4090');
  const [tcoGpuCount, setTcoGpuCount] = useState(1);

  const gpus = gpuPricingRaw as GpuInstance[];

  // Extract unique providers and architectures
  const providers = useMemo(() => {
    const set = new Set<string>();
    gpus.forEach((g) => {
      const baseProvider = g.provider.split(' ')[0];
      set.add(baseProvider);
    });
    return Array.from(set);
  }, [gpus]);

  const architectures = useMemo(() => {
    const set = new Set<string>();
    gpus.forEach((g) => set.add(g.architecture.split(' ')[0]));
    return Array.from(set);
  }, [gpus]);

  // Duration multiplier
  const durationMultiplier = useMemo(() => {
    switch (durationMode) {
      case '24h':
        return 24;
      case '100h':
        return 100;
      case '720h':
        return 720;
      default:
        return 1;
    }
  }, [durationMode]);

  // Filter and sort GPUs
  const filteredGpus = useMemo(() => {
    return gpus
      .filter((gpu) => {
        // VRAM Compatibility check
        if (onlyCompatible && gpu.totalVramGb < requiredVramGb) {
          return false;
        }

        // Search query
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matches =
            gpu.gpuModel.toLowerCase().includes(q) ||
            gpu.provider.toLowerCase().includes(q) ||
            gpu.architecture.toLowerCase().includes(q) ||
            gpu.category.toLowerCase().includes(q);
          if (!matches) return false;
        }

        // Provider filter
        if (selectedProviders.length > 0) {
          const baseProvider = gpu.provider.split(' ')[0];
          if (!selectedProviders.includes(baseProvider)) return false;
        }

        // Architecture filter
        if (selectedArchs.length > 0) {
          const baseArch = gpu.architecture.split(' ')[0];
          if (!selectedArchs.includes(baseArch)) return false;
        }

        // GPU Count filter
        if (gpuCountFilter === 'single' && gpu.gpuCount > 1) return false;
        if (gpuCountFilter === 'multi' && gpu.gpuCount <= 1) return false;

        return true;
      })
      .sort((a, b) => {
        let valA = 0;
        let valB = 0;

        if (sortField === 'costPerGb') {
          valA = a.spotPricePerHour / a.totalVramGb;
          valB = b.spotPricePerHour / b.totalVramGb;
        } else if (sortField === 'tokensPerDollar') {
          const tpsA = (a.memoryBandwidthGbps / 120) * 0.65;
          const tpsB = (b.memoryBandwidthGbps / 120) * 0.65;
          valA = calculateTokensPerDollar(a.spotPricePerHour, tpsA);
          valB = calculateTokensPerDollar(b.spotPricePerHour, tpsB);
        } else {
          valA = a[sortField] as number;
          valB = b[sortField] as number;
        }

        if (sortDirection === 'asc') {
          return valA > valB ? 1 : -1;
        } else {
          return valA < valB ? 1 : -1;
        }
      });
  }, [
    gpus,
    requiredVramGb,
    onlyCompatible,
    searchQuery,
    selectedProviders,
    selectedArchs,
    gpuCountFilter,
    sortField,
    sortDirection,
  ]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const toggleProvider = (p: string) => {
    setSelectedProviders((prev) =>
      prev.includes(p) ? prev.filter((item) => item !== p) : [...prev, p]
    );
  };

  const toggleArch = (a: string) => {
    setSelectedArchs((prev) =>
      prev.includes(a) ? prev.filter((item) => item !== a) : [...prev, a]
    );
  };

  // Best deal highlights
  const lowestSpotDeal = useMemo(() => {
    if (filteredGpus.length === 0) return null;
    return [...filteredGpus].sort((a, b) => a.spotPricePerHour - b.spotPricePerHour)[0];
  }, [filteredGpus]);

  return (
    <div className="rounded-2xl border border-slate-800/80 bg-slate-900/60 p-6 backdrop-blur-xl shadow-xl shadow-black/20 space-y-6">
      {/* Header & Quick Insights */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-slate-100">Live Cloud GPU Cost & Pricing Engine</h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-medium">
              {filteredGpus.length} Available Nodes
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time verified pricing across RunPod, Lambda Labs, Vast.ai, AWS, GCP, and specialized clouds.
          </p>
        </div>

        {/* Duration Projection Selector */}
        <div className="flex items-center gap-1.5 bg-slate-950/80 p-1.5 rounded-xl border border-slate-800 text-xs">
          <Clock className="w-4 h-4 text-slate-400 ml-1 mr-0.5" />
          <span className="text-slate-400 font-medium hidden sm:inline">Billing Span:</span>
          {[
            { id: '1h', label: '1 Hour' },
            { id: '24h', label: '24 Hours' },
            { id: '100h', label: '100h Run' },
            { id: '720h', label: '1 Month' },
          ].map((d) => (
            <button
              key={d.id}
              onClick={() => setDurationMode(d.id as DurationMode)}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                durationMode === d.id
                  ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      {/* Best Deal Spotlight & TCO Calculator Trigger */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {lowestSpotDeal && (
          <div className="md:col-span-2 rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0">
                <TrendingDown className="w-4 h-4 text-emerald-400" />
              </div>
              <div>
                <span className="text-emerald-400 font-semibold uppercase tracking-wider text-[10px] block">
                  Lowest Cost Compatible Option
                </span>
                <span className="text-slate-200 font-medium">
                  {lowestSpotDeal.gpuModel} on <b className="text-white">{lowestSpotDeal.provider}</b> ({lowestSpotDeal.totalVramGb}GB VRAM)
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="text-right">
                <span className="text-[10px] text-slate-400 block">Spot Rate</span>
                <span className="text-sm font-mono font-bold text-emerald-400">
                  ${(lowestSpotDeal.spotPricePerHour * durationMultiplier).toFixed(2)}
                  <span className="text-[10px] text-slate-400 font-normal"> / {durationMode}</span>
                </span>
              </div>
              <a
                href={lowestSpotDeal.rentUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors flex items-center gap-1"
              >
                <span>Rent Node</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        )}

        {/* TCO Buy vs Rent Trigger Card */}
        <div className="rounded-xl border border-indigo-500/30 bg-indigo-950/30 p-3.5 flex items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center shrink-0">
              <Calculator className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <span className="text-indigo-300 font-semibold text-[11px] block">
                Buy vs. Rent TCO Lab
              </span>
              <span className="text-[10px] text-slate-400">Hardware break-even simulator</span>
            </div>
          </div>
          <button
            onClick={() => setShowTcoModal(true)}
            className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-colors shrink-0"
          >
            Calculate TCO
          </button>
        </div>
      </div>

      {/* Buy vs Rent TCO Modal */}
      {showTcoModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="w-full max-w-xl bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-2xl space-y-5 text-xs">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Calculator className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-bold text-white">On-Premises Hardware vs. Cloud TCO Simulator</h3>
              </div>
              <button
                onClick={() => setShowTcoModal(false)}
                className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Selector */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-[11px] font-semibold text-slate-300 block mb-1">Target GPU Model:</label>
                <select
                  value={tcoGpuModel}
                  onChange={(e) => setTcoGpuModel(e.target.value as any)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-slate-200"
                >
                  <option value="rtx4090">NVIDIA RTX 4090 (24GB) — $1,599 MSRP</option>
                  <option value="rtx5090">NVIDIA RTX 5090 (32GB) — $1,999 MSRP</option>
                  <option value="a100">NVIDIA A100 (80GB PCIe) — $9,500 Market</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-300 block mb-1">GPU Quantity:</label>
                <div className="flex gap-2">
                  {[1, 2, 4].map((count) => (
                    <button
                      key={count}
                      onClick={() => setTcoGpuCount(count)}
                      className={`flex-1 py-2 font-mono font-bold rounded-xl border ${
                        tcoGpuCount === count
                          ? 'bg-indigo-600 border-indigo-400 text-white'
                          : 'bg-slate-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      {count}x GPU
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* TCO Results Grid */}
            {(() => {
              const msrp = tcoGpuModel === 'rtx4090' ? 1599 : tcoGpuModel === 'rtx5090' ? 1999 : 9500;
              const watts = tcoGpuModel === 'rtx4090' ? 450 : tcoGpuModel === 'rtx5090' ? 600 : 400;
              const cloudRate = tcoGpuModel === 'rtx4090' ? 0.44 : tcoGpuModel === 'rtx5090' ? 0.79 : 1.64;
              const res = calculateTcoBreakEven(tcoGpuCount, msrp, 1200, watts, 0.14, cloudRate);

              return (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">Upfront Hardware</span>
                      <span className="text-base font-bold font-mono text-indigo-400">
                        ${res.hardwareUpfrontCost.toLocaleString()}
                      </span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">Monthly Electricity</span>
                      <span className="text-base font-bold font-mono text-amber-400">
                        ${res.monthlyElectricityCost}/mo
                      </span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">Cloud Rent (Monthly)</span>
                      <span className="text-base font-bold font-mono text-rose-400">
                        ${res.monthlyCloudCost.toLocaleString()}/mo
                      </span>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/40 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-semibold text-emerald-300">Break-Even Horizon:</span>
                      <span className="text-lg font-mono font-black text-emerald-400">
                        {res.breakEvenMonths} Months
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-relaxed">
                      If running 24/7 workloads, purchasing hardware pays for itself after <b>{res.breakEvenMonths} months</b>.
                      Estimated 1-year net savings: <b className="text-emerald-300">${res.savings1Year.toLocaleString()}</b> (2-year: <b className="text-emerald-300">${res.savings2Year.toLocaleString()}</b>).
                    </p>
                  </div>
                </div>
              );
            })()}

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setShowTcoModal(false)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-semibold"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Controls and Filter Bar */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          {/* Search bar */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search GPU model, provider, architecture..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>

          {/* Compatibility Toggle */}
          <button
            onClick={() => setOnlyCompatible(!onlyCompatible)}
            className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 border transition-all ${
              onlyCompatible
                ? 'bg-indigo-950/60 border-indigo-500 text-indigo-200'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <CheckCircle className={`w-3.5 h-3.5 ${onlyCompatible ? 'text-indigo-400' : 'text-slate-500'}`} />
            <span>≥ {requiredVramGb} GB VRAM Only</span>
          </button>

          {/* GPU Count Filter */}
          <div className="flex items-center rounded-xl bg-slate-950 border border-slate-800 p-1 text-xs">
            {(['all', 'single', 'multi'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setGpuCountFilter(mode)}
                className={`px-2.5 py-1 rounded-lg capitalize transition-all ${
                  gpuCountFilter === mode
                    ? 'bg-slate-800 text-slate-100 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {mode === 'all' ? 'All Setup' : mode === 'single' ? '1x GPU' : 'Multi-GPU'}
              </button>
            ))}
          </div>
        </div>

        {/* Provider and Architecture Chips */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-[11px] text-slate-400 flex items-center gap-1 mr-1">
            <Filter className="w-3 h-3 text-slate-500" />
            Providers:
          </span>
          {providers.map((p) => {
            const isSelected = selectedProviders.includes(p);
            return (
              <button
                key={p}
                onClick={() => toggleProvider(p)}
                className={`text-[11px] px-2.5 py-0.5 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-indigo-600/30 text-indigo-200 border-indigo-500/60'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-300'
                }`}
              >
                {p}
              </button>
            );
          })}

          <div className="w-px h-4 bg-slate-800 mx-1 hidden sm:block" />

          <span className="text-[11px] text-slate-400 mr-1 hidden sm:inline">Arch:</span>
          {architectures.map((a) => {
            const isSelected = selectedArchs.includes(a);
            return (
              <button
                key={a}
                onClick={() => toggleArch(a)}
                className={`text-[11px] px-2.5 py-0.5 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-violet-600/30 text-violet-200 border-violet-500/60'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-300'
                }`}
              >
                {a}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Table Container with Scrollable Body & Sticky Header */}
      <div className="overflow-x-auto overflow-y-auto max-h-[540px] rounded-xl border border-slate-800 bg-slate-950/70 relative">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="sticky top-0 z-20 bg-slate-900/95 backdrop-blur-md shadow-sm">
            <tr className="border-b border-slate-800 text-slate-300 font-semibold select-none">
              <th className="py-3 px-4">GPU & Architecture</th>
              <th className="py-3 px-3">Provider</th>
              <th
                onClick={() => handleSort('totalVramGb')}
                className="py-3 px-3 cursor-pointer hover:text-indigo-400 transition-colors"
              >
                <div className="flex items-center gap-1">
                  <span>Total VRAM</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>
              <th
                onClick={() => handleSort('memoryBandwidthGbps')}
                className="py-3 px-3 cursor-pointer hover:text-indigo-400 transition-colors hidden md:table-cell"
              >
                <div className="flex items-center gap-1">
                  <span>Memory Bandwidth</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>
              <th
                onClick={() => handleSort('spotPricePerHour')}
                className="py-3 px-3 cursor-pointer hover:text-indigo-400 transition-colors text-right"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Spot ({durationMode})</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>
              <th
                onClick={() => handleSort('onDemandPricePerHour')}
                className="py-3 px-3 cursor-pointer hover:text-indigo-400 transition-colors text-right"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>On-Demand ({durationMode})</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>
              <th
                className="py-3 px-3 hover:text-indigo-400 transition-colors text-right hidden lg:table-cell"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Throughput ROI</span>
                  <Zap className="w-3 h-3 text-cyan-400" />
                </div>
              </th>
              <th className="py-3 px-4 text-center">Action</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-850">
            {filteredGpus.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-12 text-center text-slate-500">
                  <div className="max-w-xs mx-auto space-y-2">
                    <Server className="w-8 h-8 text-slate-600 mx-auto" />
                    <p className="font-medium text-slate-400">No matching GPU nodes found</p>
                    <p className="text-[11px] text-slate-500">
                      Try unchecking the VRAM filter or clearing provider filters.
                    </p>
                    <button
                      onClick={() => {
                        setOnlyCompatible(false);
                        setSelectedProviders([]);
                        setSelectedArchs([]);
                        setSearchQuery('');
                      }}
                      className="mt-2 px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs"
                    >
                      Reset All Filters
                    </button>
                  </div>
                </td>
              </tr>
            ) : (
              filteredGpus.map((gpu) => {
                const isCompatible = gpu.totalVramGb >= requiredVramGb;
                const spotCost = gpu.spotPricePerHour * durationMultiplier;
                const onDemandCost = gpu.onDemandPricePerHour * durationMultiplier;
                const effectiveTps = Math.max(1, (gpu.memoryBandwidthGbps / 120) * 0.65);
                const tokensPerDollar = calculateTokensPerDollar(gpu.spotPricePerHour, effectiveTps);
                const costPer1M = calculateCostPerMillionTokens(gpu.spotPricePerHour, effectiveTps);

                return (
                  <tr
                    key={gpu.id}
                    className="hover:bg-slate-900/60 transition-colors group"
                  >
                    {/* GPU & Specs */}
                    <td className="py-3 px-4">
                      <div className="flex items-start gap-2.5">
                        <div className="w-7 h-7 rounded-lg bg-slate-800/80 border border-slate-700 flex items-center justify-center shrink-0 mt-0.5">
                          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                        </div>
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-bold text-slate-100">{gpu.gpuModel}</span>
                            {gpu.popular && (
                              <span className="text-[9px] px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 font-semibold">
                                Popular
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-0.5">
                            <span className="text-slate-300">{gpu.architecture}</span>
                            <span>•</span>
                            <span>{gpu.interconnect}</span>
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* Provider */}
                    <td className="py-3 px-3">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-slate-800/80 text-slate-300 border border-slate-700/60">
                        {gpu.provider}
                      </span>
                    </td>

                    {/* VRAM */}
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`font-mono font-bold text-xs ${
                            isCompatible ? 'text-emerald-400' : 'text-slate-400'
                          }`}
                        >
                          {gpu.totalVramGb} GB
                        </span>
                        {gpu.gpuCount > 1 && (
                          <span className="text-[10px] text-slate-500 font-mono">
                            ({gpu.gpuCount}x {gpu.vramPerGpuGb}GB)
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Bandwidth */}
                    <td className="py-3 px-3 hidden md:table-cell">
                      <span className="font-mono text-slate-300 text-xs">
                        {gpu.memoryBandwidthGbps.toLocaleString()} GB/s
                      </span>
                    </td>

                    {/* Spot Price */}
                    <td className="py-3 px-3 text-right">
                      <span className="font-mono font-bold text-emerald-400 text-xs">
                        ${spotCost >= 100 ? spotCost.toFixed(1) : spotCost.toFixed(2)}
                      </span>
                      <span className="block text-[10px] text-slate-500">
                        (${gpu.spotPricePerHour.toFixed(2)}/hr)
                      </span>
                    </td>

                    {/* On-Demand Price */}
                    <td className="py-3 px-3 text-right">
                      <span className="font-mono font-bold text-slate-200 text-xs">
                        ${onDemandCost >= 100 ? onDemandCost.toFixed(1) : onDemandCost.toFixed(2)}
                      </span>
                      <span className="block text-[10px] text-slate-500">
                        (${gpu.onDemandPricePerHour.toFixed(2)}/hr)
                      </span>
                    </td>

                    {/* ROI (Tokens / $) */}
                    <td className="py-3 px-3 text-right hidden lg:table-cell">
                      <span className="font-mono font-bold text-cyan-300 text-xs">
                        {tokensPerDollar} MTok / $
                      </span>
                      <span className="block text-[10px] text-slate-400 font-mono">
                        ${costPer1M} / 1M
                      </span>
                    </td>

                    {/* Action Button */}
                    <td className="py-3 px-4 text-center">
                      <a
                        href={gpu.rentUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center justify-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-600/80 hover:bg-indigo-600 text-white font-semibold text-xs transition-all shadow-sm hover:shadow-indigo-500/20 group-hover:scale-105"
                      >
                        <span>Rent</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Footer / Transparency Note */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500 pt-2">
        <div className="flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>Showing all {filteredGpus.length} nodes • Scroll table vertically to view full hardware catalog</span>
        </div>
        <span>Prices updated February 2026</span>
      </div>
    </div>
  );
};
