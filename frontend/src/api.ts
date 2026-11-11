import type { OverviewData, WorklistMember, MemberTwin, MemberDetail, AgentStep, ApprovalItem } from './types';

const API_BASE = '';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchOverview(): Promise<OverviewData> {
  const res = await fetch(`${API_BASE}/v1/overview`);
  return res.json();
}

export async function fetchWorklist(capacity = 50, strategy = 'eac'): Promise<{ members: WorklistMember[]; disclaimer: string }> {
  const res = await fetch(`${API_BASE}/v1/worklist?capacity=${capacity}&strategy=${strategy}`);
  return res.json();
}

export async function fetchMemberDetail(id: string): Promise<MemberDetail> {
  const res = await fetch(`${API_BASE}/v1/members/${id}`);
  return res.json();
}

export async function fetchMemberTwin(id: string): Promise<MemberTwin> {
  const res = await fetch(`${API_BASE}/v1/members/${id}/twin`);
  return res.json();
}

export async function fetchMemberTimeline(id: string) {
  const res = await fetch(`${API_BASE}/v1/members/${id}/timeline`);
  return res.json();
}

export async function fetchAgentJournal(limit = 40): Promise<{ steps: AgentStep[] }> {
  const res = await fetch(`${API_BASE}/v1/agents/journal?limit=${limit}`);
  return res.json();
}

export async function fetchApprovals(status = 'pending'): Promise<ApprovalItem[]> {
  const res = await fetch(`${API_BASE}/v1/approvals?status=${status}`);
  return res.json();
}

export async function postReview(id: string, approve: boolean, reviewer: string, note = '') {
  const res = await fetch(`${API_BASE}/v1/approvals/${id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approve, reviewer, note }),
  });
  return res.json();
}

export async function fetchReport(name: string) {
  const res = await fetch(`${API_BASE}/v1/reports/${name}`);
  return res.json();
}

export async function fetchStreamingReport() {
  const res = await fetch(`${API_BASE}/v1/streaming/report`);
  return res.json();
}
