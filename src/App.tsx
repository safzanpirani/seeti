import { useEffect, useRef, useState } from 'react';
import { WhistleDetector, type DetectorState } from './detector';

const IDLE_STATE: DetectorState = {
  running: false,
  whistling: false,
  count: 0,
  peakFreq: 0,
  peakDb: -120,
  prominenceDb: 0,
  holdMs: 0,
};

function beep(times: number) {
  const ctx = new AudioContext();
  for (let i = 0; i < times; i++) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    const t = ctx.currentTime + i * 0.35;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.5, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.28);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.3);
  }
  setTimeout(() => ctx.close(), times * 350 + 500);
}

export default function App() {
  const [state, setState] = useState<DetectorState>(IDLE_STATE);
  const [target, setTarget] = useState(3);
  const [sensitivity, setSensitivity] = useState(0.6);
  const [done, setDone] = useState(false);
  const [error, setError] = useState('');
  const detector = useRef<WhistleDetector | null>(null);
  const wakeLock = useRef<WakeLockSentinel | null>(null);
  const targetRef = useRef(target);
  targetRef.current = target;

  useEffect(() => () => detector.current?.stop(), []);

  async function start() {
    setError('');
    setDone(false);
    const d = new WhistleDetector({
      sensitivity,
      onWhistleStart: () => {
        navigator.vibrate?.(200);
        beep(1);
      },
      onWhistleEnd: (count) => {
        if (count >= targetRef.current) {
          setDone(true);
          navigator.vibrate?.([400, 150, 400, 150, 400]);
          beep(6);
        }
      },
      onFrame: setState,
    });
    try {
      await d.start();
      detector.current = d;
      try {
        wakeLock.current = (await navigator.wakeLock?.request('screen')) ?? null;
      } catch {
        /* wake lock is best-effort */
      }
    } catch {
      setError('Microphone access denied — allow the mic permission and try again.');
    }
  }

  function stop() {
    detector.current?.stop();
    detector.current = null;
    wakeLock.current?.release();
    wakeLock.current = null;
  }

  function reset() {
    detector.current?.resetCount();
    setDone(false);
    setState((s) => ({ ...s, count: 0 }));
  }

  const { running, whistling, count } = state;
  const levelPct = Math.max(0, Math.min(100, ((state.peakDb + 90) / 70) * 100));
  const armed = state.holdMs > 0 && !whistling;

  return (
    <main className={`shell ${whistling ? 'is-whistling' : ''} ${done ? 'is-done' : ''}`}>
      <div className="steam" aria-hidden="true">
        <span /><span /><span />
      </div>

      <header>
        <h1>SEETI</h1>
        <p className="tagline">pressure-cooker whistle counter</p>
      </header>

      <section className="gauge" aria-live="polite">
        <div className="count-ring">
          <span className="count">{count}</span>
          <span className="of">of {target}</span>
        </div>
        <p className="status">
          {done
            ? 'done — turn off the stove!'
            : whistling
              ? `whistling · ${Math.round(state.peakFreq)} Hz`
              : armed
                ? 'hearing something…'
                : running
                  ? 'listening'
                  : 'paused'}
        </p>
        <div className="meter" aria-hidden="true">
          <div className="meter-fill" style={{ width: running ? `${levelPct}%` : '0%' }} />
        </div>
      </section>

      <section className="controls">
        <label className="row">
          <span>whistles</span>
          <div className="stepper">
            <button onClick={() => setTarget((t) => Math.max(1, t - 1))} aria-label="fewer">−</button>
            <b>{target}</b>
            <button onClick={() => setTarget((t) => Math.min(20, t + 1))} aria-label="more">+</button>
          </div>
        </label>

        <label className="row">
          <span>sensitivity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={sensitivity}
            onChange={(e) => {
              const v = Number(e.target.value);
              setSensitivity(v);
              detector.current?.setSensitivity(v);
            }}
          />
        </label>

        <div className="buttons">
          {!running ? (
            <button className="primary" onClick={start}>start listening</button>
          ) : (
            <button className="primary stop" onClick={stop}>stop</button>
          )}
          <button className="ghost" onClick={reset}>reset count</button>
        </div>

        {error && <p className="error">{error}</p>}
        <p className="hint">
          keep the phone near the cooker · screen stays awake while listening ·
          talking won't trigger it
        </p>
      </section>
    </main>
  );
}
