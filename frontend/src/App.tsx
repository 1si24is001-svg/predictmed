import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { OverviewView } from './components/OverviewView';
import { Member360View } from './components/Member360View';
import { WorklistView } from './components/WorklistView';
import { EvidenceView } from './components/EvidenceView';
import { AgentsView } from './components/AgentsView';
import { StreamingView } from './components/StreamingView';
import { GovernanceView } from './components/GovernanceView';
import { fetchHealth, fetchOverview } from './api';
import type { OverviewData } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [online, setOnline] = useState(false);
  const [overviewData, setOverviewData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      try {
        const h = await fetchHealth();
        setOnline(h.status === 'ok');
        const ov = await fetchOverview();
        setOverviewData(ov);
      } catch (e) {
        console.error('Failed to initialize connection with PredictMed API:', e);
        setOnline(false);
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        online={online}
        cohortCount={overviewData?.total_members || 22664}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'overview' && <OverviewView data={overviewData} loading={loading} />}
        {activeTab === 'member360' && <Member360View />}
        {activeTab === 'worklist' && <WorklistView />}
        {activeTab === 'evidence' && <EvidenceView />}
        {activeTab === 'agents' && <AgentsView />}
        {activeTab === 'streaming' && <StreamingView />}
        {activeTab === 'governance' && <GovernanceView />}
      </main>

      <footer className="bg-white border-t border-slate-200 py-4 text-center text-xs text-slate-400">
        PredictMed Clinical Intelligence Platform &copy; 2026 &bull; Research Prototype &bull; HIL Safeguarded
      </footer>
    </div>
  );
}

export default App;
