export interface OverviewData {
  total_members: number;
  mean_risk: number;
  total_eac_usd: number;
  high_risk_count: number;
  high_risk_threshold: number;
  plan_distribution: Record<string, number>;
  kdigo_stages: Record<string, number>;
  risk_histogram: { bin: string; count: number }[];
  disclaimer: string;
}

export interface WorklistMember {
  member_id: string;
  rank: number;
  p_deterioration: number;
  va_low?: number;
  va_high?: number;
  conformal_label?: string;
  route?: string;
  care_gaps?: string;
  modifiable_fraction: number;
  eac_usd: number;
  twin_runway_months?: number;
  plan_type?: string;
}

export interface MemberTwin {
  member_id: string;
  current_egfr: number;
  slope_per_year: number;
  runway_months: number;
  projection_months: number[];
  natural_trajectory: number[];
  counterfactual_trajectory: number[];
  disclaimer: string;
}

export interface MemberDetail {
  member_id: string;
  as_of: string;
  estimated_probability: number;
  probability_interval: [number, number];
  conformal_label: string;
  p_crash_dialysis_24m: number;
  care_gaps: string[];
  modifiable_fraction: number;
  expected_avoidable_cost_usd: number;
  top_reasons: { text: string; feature?: string; value?: number }[];
  twin: Record<string, any>;
  disclaimer: string;
}

export interface AgentStep {
  step_id: string;
  ts: string;
  agent: string;
  kind: string;
  message: string;
  evidence: any;
  decision?: string;
}

export interface ApprovalItem {
  id: string;
  ts: string;
  kind: string;
  summary: string;
  status: "pending" | "approved" | "rejected";
  details: any;
}

