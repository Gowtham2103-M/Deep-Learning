import { useCallback, useRef } from "react";

/**
 * Text-to-speech using the browser's built-in speechSynthesis (free, works offline on Windows).
 * Auto-speaking is rate limited: the same sentence is not repeated during the cooldown.
 */
export function useSpeak(cooldownSeconds: number) {
  const last = useRef<{ text: string; at: number }>({ text: "", at: 0 });

  const say = useCallback((text: string) => {
    if (!("speechSynthesis" in window) || !text) return false;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-IN";
    window.speechSynthesis.speak(u);
    last.current = { text, at: Date.now() };
    return true;
  }, []);

  /** Manual button: always speaks. */
  const speak = useCallback((text: string) => say(text), [say]);

  /** Automatic speech after a recognition: debounced. */
  const speakAuto = useCallback(
    (text: string) => {
      const { text: prev, at } = last.current;
      const elapsed = Date.now() - at;
      if (text === prev && elapsed < cooldownSeconds * 1000) return false; // same sentence again
      if (elapsed < 1000) return false;                                    // anything too soon
      return say(text);
    },
    [cooldownSeconds, say]
  );

  return { speak, speakAuto, supported: typeof window !== "undefined" && "speechSynthesis" in window };
}
