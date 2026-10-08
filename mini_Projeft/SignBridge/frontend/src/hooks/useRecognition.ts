import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, AppConfig, PredictResponse, predictFrames } from "../api";

/**
 * Camera -> frame buffer -> backend prediction loop.
 *
 *  - every 1/capture_fps seconds one JPEG frame is added to a sliding window buffer
 *  - as soon as the window is (nearly) full and no request is running, the frames are sent
 *  - a sentence is only "recognized" when the same accepted prediction repeats `stable_count` times
 *
 * Latency is MEASURED (server inference time and full round trip); nothing is assumed.
 */
export function useRecognition(cfg: AppConfig | null, onCommit: (text: string) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const bufferRef = useRef<Blob[]>([]);
  const runningRef = useRef(false);
  const recentRef = useRef<string[]>([]);
  const cbRef = useRef(onCommit);
  cbRef.current = onCommit;

  const [running, setRunning] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [recognized, setRecognized] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [roundTripMs, setRoundTripMs] = useState<number | null>(null);
  const [bufferFill, setBufferFill] = useState(0);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    bufferRef.current = [];
    recentRef.current = [];
    setBufferFill(0);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    if (!cfg || runningRef.current) return;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
    } catch {
      setError("Cannot open the camera. Allow camera permission and make sure no other app is using it.");
      return;
    }
    runningRef.current = true;
    setRunning(true);

    const canvas = document.createElement("canvas");
    const maxFrames = Math.max(2, Math.round(cfg.window_seconds * cfg.capture_fps));

    // ---- capture loop: one frame every 1/capture_fps seconds
    const capture = async () => {
      while (runningRef.current) {
        const t0 = performance.now();
        const v = videoRef.current;
        if (v && v.readyState >= 2 && v.videoWidth > 0) {
          canvas.width = 320;
          canvas.height = Math.round((320 * v.videoHeight) / v.videoWidth);
          canvas.getContext("2d")!.drawImage(v, 0, 0, canvas.width, canvas.height);
          const blob: Blob | null = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.8));
          if (blob) {
            bufferRef.current.push(blob);
            if (bufferRef.current.length > maxFrames) bufferRef.current.shift();
            setBufferFill(bufferRef.current.length / maxFrames);
          }
        }
        const wait = Math.max(0, 1000 / cfg.capture_fps - (performance.now() - t0));
        await new Promise((r) => setTimeout(r, wait));
      }
    };

    // ---- inference loop: never sends a new request while one is still running
    const infer = async () => {
      while (runningRef.current) {
        if (bufferRef.current.length < Math.max(2, Math.floor(maxFrames * 0.9))) {
          await new Promise((r) => setTimeout(r, 200));
          continue;
        }
        const frames = [...bufferRef.current];
        const t0 = performance.now();
        try {
          const res = await predictFrames(frames);
          setRoundTripMs(performance.now() - t0);
          setPrediction(res);
          if (res.accepted) {
            recentRef.current = [...recentRef.current, res.text].slice(-cfg.stable_count);
            const r = recentRef.current;
            if (r.length === cfg.stable_count && r.every((x) => x === r[0])) {
              setRecognized(r[0]);
              cbRef.current(r[0]);
            }
          } else {
            recentRef.current = [];
          }
          setError(null);
        } catch (e) {
          if (e instanceof ApiError && e.code === "model_not_trained") {
            setError(e.message);
            stop();
            return;
          }
          setError(e instanceof Error ? e.message : "Prediction failed.");
          await new Promise((r) => setTimeout(r, 1500));
        }
      }
    };

    void capture();
    void infer();
  }, [cfg, stop]);

  return { videoRef, running, start, stop, prediction, recognized, setRecognized, error, roundTripMs, bufferFill };
}
