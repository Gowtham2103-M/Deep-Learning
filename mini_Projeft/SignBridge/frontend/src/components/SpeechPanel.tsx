import { useRef, useState } from "react";
import { ApiError, MatchResponse, matchSpeech, referenceVideoUrl } from "../api";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";

/** Person A: speech -> text -> ISL reference video */
export default function SpeechPanel() {
  const [result, setResult] = useState<MatchResponse | null>(null);
  const [heard, setHeard] = useState("");
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const doMatch = async (text: string) => {
    setHeard(text);
    setError(null);
    try {
      setResult(await matchSpeech(text));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.message : "Matching failed.");
    }
  };

  const mic = useSpeechRecognition(doMatch);
  const m = result?.match;

  return (
    <section className="card">
      <h2>Speech &rarr; Text &rarr; ISL</h2>
      <div className="row">
        {!mic.listening ? (
          <button className="btn primary" onClick={mic.start} disabled={!mic.supported}>Start Microphone</button>
        ) : (
          <button className="btn danger" onClick={mic.stop}>Stop</button>
        )}
        {!mic.supported && (
          <span className="muted">Speech input needs Chrome or Edge. You can type the sentence instead.</span>
        )}
      </div>
      {mic.error && <div className="banner banner-error">{mic.error}</div>}

      <div className="row">
        <input
          className="text-input"
          placeholder="...or type a sentence"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && typed.trim() && doMatch(typed.trim())}
        />
        <button className="btn" disabled={!typed.trim()} onClick={() => doMatch(typed.trim())}>Match</button>
      </div>

      <h3>Recognized Speech</h3>
      <div className="big-text">{(mic.listening ? mic.transcript : heard) || "-"}</div>

      {error && <div className="banner banner-error">{error}</div>}
      {result?.status === "dataset_missing" && <div className="banner banner-warn">{result.message}</div>}
      {result?.status === "no_match" && (
        <div className="banner banner-warn">{result.message}</div>
      )}
      {m && (
        <div className="match">
          <div>
            Matched sentence: <strong>{m.label.toUpperCase()}</strong>{" "}
            <span className="muted">({m.method} match, score {m.score})</span>
          </div>
          <video ref={videoRef} key={m.video_url} src={referenceVideoUrl(m.video_url)} controls playsInline />
          <button className="btn primary" onClick={() => { const v = videoRef.current; if (v) { v.currentTime = 0; void v.play(); } }}>
            Play ISL Sign
          </button>
          <p className="muted">Reference sign video taken from the dataset (first video of this sentence).</p>
        </div>
      )}
    </section>
  );
}
