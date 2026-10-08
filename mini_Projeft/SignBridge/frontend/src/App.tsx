import { useCallback, useEffect, useState } from "react";
import { getStatus, StatusResponse } from "./api";
import MetricsPanel from "./components/MetricsPanel";
import RecognitionPanel from "./components/RecognitionPanel";
import SpeechPanel from "./components/SpeechPanel";
import StatusBanner from "./components/StatusBanner";

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const load = useCallback(() => {
    getStatus()
      .then((s) => { setStatus(s); setError(null); })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="page">
      <header>
        <h1>SIGNBRIDGE</h1>
        <p>Two-Way ISL Communication Assistant</p>
        <button className="btn small" onClick={() => { load(); setRefreshKey((k) => k + 1); }}>
          Refresh status
        </button>
      </header>

      <StatusBanner status={status} error={error} />

      <RecognitionPanel cfg={status?.config ?? null} modelTrained={!!status?.model.trained} />
      <SpeechPanel />
      <MetricsPanel refreshKey={refreshKey} />

      <footer>
        Prototype: 100-class sentence-level closed-set ISL classification. It is not unrestricted continuous ISL translation.
      </footer>
    </div>
  );
}
