import { useState } from "react";
import { AppConfig } from "../api";
import { useRecognition } from "../hooks/useRecognition";
import { useSpeak } from "../hooks/useSpeak";

interface Props {
  cfg: AppConfig | null;
  modelTrained: boolean;
}

/** Person B: ISL (camera) -> text -> speech */
export default function RecognitionPanel({ cfg, modelTrained }: Props) {
  const { speak, speakAuto, supported } = useSpeak(cfg?.speak_cooldown_seconds ?? 4);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const rec = useRecognition(cfg, (text) => {
    if (autoSpeak) speakAuto(text);
  });
  const p = rec.prediction;

  const pct = p ? Math.round(p.confidence * 100) : 0;
  return (
    <section className="card">
      <h2>ISL &rarr; Text &rarr; Speech</h2>
      <div className="grid-2">
        <div>
          <div className="video-wrap">
            <video ref={rec.videoRef} muted playsInline />
            {!rec.running && <div className="video-overlay">Camera is off</div>}
          </div>
          <div className="row">
            {!rec.running ? (
              <button className="btn primary" onClick={rec.start} disabled={!cfg || !modelTrained}>
                Start recognition
              </button>
            ) : (
              <button className="btn danger" onClick={rec.stop}>Stop recognition</button>
            )}
            <label className="check">
              <input type="checkbox" checked={autoSpeak} onChange={(e) => setAutoSpeak(e.target.checked)} />
              Auto-speak
            </label>
          </div>
          {!modelTrained && <p className="muted">Recognition is disabled until the model is trained.</p>}
          {rec.running && (
            <div className="bar" title="Frame window filling">
              <div style={{ width: `${Math.round(rec.bufferFill * 100)}%` }} />
            </div>
          )}
        </div>

        <div className="prediction">
          <h3>Prediction</h3>
          {p ? (
            <>
              <div className={`pred-label ${p.accepted ? "" : "faded"}`}>{p.label.toUpperCase()}</div>
              <div className="muted">
                Confidence: <strong>{pct}%</strong>{" "}
                {p.accepted ? "" : `(below the ${Math.round(p.threshold * 100)}% threshold - not accepted)`}
              </div>
              <ul className="topk">
                {p.top_k.map((t) => (
                  <li key={t.class_id}>{t.label} <span>{(t.confidence * 100).toFixed(1)}%</span></li>
                ))}
              </ul>
              <div className="latency">
                <div>Inference (server): <strong>{p.latency_ms.total_ms.toFixed(0)} ms</strong> on {p.device}</div>
                <div>Round trip (measured): <strong>{rec.roundTripMs?.toFixed(0) ?? "-"} ms</strong></div>
                <div className="muted">
                  Each prediction uses the last {cfg?.window_seconds}s of video, so a sign is recognised only after it has been performed.
                </div>
              </div>
            </>
          ) : (
            <p className="muted">{rec.running ? "Collecting frames..." : "No prediction yet."}</p>
          )}
        </div>
      </div>

      {rec.error && <div className="banner banner-error">{rec.error}</div>}

      <div className="recognized">
        <h3>Recognized Text</h3>
        <div className="big-text">{rec.recognized ? rec.recognized.toUpperCase() : "-"}</div>
        <button className="btn" disabled={!rec.recognized || !supported} onClick={() => speak(rec.recognized)}>
          Speak
        </button>
        {!supported && <span className="muted"> Text-to-speech is not supported in this browser.</span>}
      </div>
    </section>
  );
}
