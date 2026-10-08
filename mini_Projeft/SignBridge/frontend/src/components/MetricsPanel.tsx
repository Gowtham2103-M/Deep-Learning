import { useEffect, useState } from "react";
import { getMetrics, MetricsResponse } from "../api";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** Shows ONLY real metrics saved by the training / evaluation scripts. */
export default function MetricsPanel({ refreshKey }: { refreshKey: number }) {
  const [data, setData] = useState<MetricsResponse | null>(null);
  useEffect(() => { getMetrics().then(setData).catch(() => setData(null)); }, [refreshKey]);

  return (
    <section className="card">
      <h2>Evaluation &amp; Performance</h2>
      {!data && <p className="muted">Metrics unavailable (backend offline).</p>}
      {data && !data.trained && <p><strong>{data.message}</strong></p>}
      {data?.trained && !data.metrics && <p>{data.message}</p>}
      {data?.metrics && (
        <>
          <div className="metrics">
            <div><span>Accuracy</span><strong>{pct(data.metrics.accuracy)}</strong></div>
            <div><span>Precision (macro)</span><strong>{pct(data.metrics.precision_macro)}</strong></div>
            <div><span>Recall (macro)</span><strong>{pct(data.metrics.recall_macro)}</strong></div>
            <div><span>F1 (macro)</span><strong>{pct(data.metrics.f1_macro)}</strong></div>
          </div>
          <p className="muted">
            Test split: {data.metrics.num_samples} videos - split mode: {data.metrics.split_mode} - evaluated {data.metrics.evaluated_at}
          </p>
          {data.metrics.split_warnings.map((w) => <div key={w} className="banner banner-info">{w}</div>)}
        </>
      )}
      {data?.trained && (
        data.latency ? (
          <p>
            Measured inference latency ({data.latency.videos} videos, {data.latency.device}): mean{" "}
            <strong>{data.latency.total_ms.mean.toFixed(0)} ms</strong>, median {data.latency.total_ms.median.toFixed(0)} ms,
            p95 {data.latency.total_ms.p95.toFixed(0)} ms.
          </p>
        ) : (
          <p className="muted">Latency not measured yet. Run: python -m training.benchmark_latency</p>
        )
      )}
    </section>
  );
}
