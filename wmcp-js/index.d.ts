export class WMCPClient {
  constructor(baseUrl?: string);
  health(): Promise<{ status: string; protocol_loaded: boolean; requests_served: number; uptime_s: number }>;
  communicate(): Promise<{ tokens_a: number[]; tokens_b: number[]; prediction: number; a_greater: boolean; latency_ms: number }>;
  validate(): Promise<{ status: string; protocol_info: object }>;
  metrics(): Promise<string>;
  dashboard(): Promise<string>;
}
