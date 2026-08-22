export interface GpuInstance {
  id: string;
  provider: string;
  providerIcon: string;
  gpuModel: string;
  architecture: string;
  gpuCount: number;
  vramPerGpuGb: number;
  totalVramGb: number;
  memoryBandwidthGbps: number;
  interconnect: string;
  spotPricePerHour: number;
  onDemandPricePerHour: number;
  monthlyEstimated: number;
  availability: 'High' | 'Medium' | 'Limited / Preview';
  category: string;
  rentUrl: string;
  popular?: boolean;
}
