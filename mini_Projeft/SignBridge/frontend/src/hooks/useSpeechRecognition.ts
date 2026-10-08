import { useCallback, useEffect, useRef, useState } from "react";

// The browser's Web Speech API (Chrome / Edge). Not typed in TypeScript's DOM lib, so we use `any`.
// Note: Chrome sends the audio to Google's servers for recognition, so it needs internet.
/* eslint-disable @typescript-eslint/no-explicit-any */
const Ctor: any =
  typeof window !== "undefined" ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition : null;

export function useSpeechRecognition(onFinal: (text: string) => void) {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<any>(null);
  const finalRef = useRef("");
  const cbRef = useRef(onFinal);
  cbRef.current = onFinal;

  const start = useCallback(() => {
    if (!Ctor) return;
    setError(null);
    setTranscript("");
    finalRef.current = "";
    const rec = new Ctor();
    rec.lang = "en-IN";
    rec.interimResults = true;
    rec.continuous = false;
    rec.onresult = (e: any) => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      setTranscript(text);
      finalRef.current = text;
    };
    rec.onerror = (e: any) => {
      const map: Record<string, string> = {
        "not-allowed": "Microphone permission was denied. Allow it in the browser address bar.",
        "no-speech": "No speech was heard. Try again.",
        network: "Speech recognition needs an internet connection in this browser.",
      };
      setError(map[e.error] ?? `Speech recognition error: ${e.error}`);
    };
    rec.onend = () => {
      setListening(false);
      if (finalRef.current.trim()) cbRef.current(finalRef.current.trim());
    };
    recRef.current = rec;
    try {
      rec.start();
      setListening(true);
    } catch {
      setError("Could not start the microphone.");
    }
  }, []);

  const stop = useCallback(() => recRef.current?.stop(), []);
  useEffect(() => () => recRef.current?.abort?.(), []);

  return { supported: !!Ctor, listening, transcript, error, start, stop };
}
