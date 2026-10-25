// Offline detector harness: synthesizes kitchen audio scenarios, emulates the
// browser AnalyserNode (Blackman window, |X|/N, time smoothing, dB) and runs
// the real WhistleCore from src/detector.ts against it.
//
//   bun scripts/harness.ts [sensitivity=0.6]
//
// Exits non-zero if any scenario's count differs from what's expected.
import { WhistleCore, thresholdDbFor } from '../src/detector';

const SR = 48000;
const FFT = 8192; // matches analyser.fftSize in detector.ts
const SMOOTHING = 0.4; // matches analyser.smoothingTimeConstant
const FPS = 60; // requestAnimationFrame cadence
const sensitivity = Number(process.argv[2] ?? 0.6);

// Deterministic PRNG so runs are reproducible.
let seed = 1234567;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const dbToAmp = (db: number) => 10 ** (db / 20);

// ---- signal generators (all add into `buf`) ----

function addNoise(buf: Float32Array, db: number) {
  // gently low-passed white noise ≈ room tone
  const a = dbToAmp(db);
  let prev = 0;
  for (let i = 0; i < buf.length; i++) {
    prev = 0.7 * prev + 0.3 * gauss();
    buf[i] += a * prev;
  }
}

/** Cooker whistle: narrowband tone with vibrato, a 2nd harmonic, breathy noise, soft attack/decay. */
function addWhistle(buf: Float32Array, startS: number, durS: number, hz: number, db: number) {
  const a = dbToAmp(db);
  const s0 = Math.floor(startS * SR);
  const n = Math.floor(durS * SR);
  let phase = 0;
  for (let i = 0; i < n && s0 + i < buf.length; i++) {
    const t = i / SR;
    const env = Math.min(1, t / 0.15, (durS - t) / 0.2);
    const f = hz + 25 * Math.sin(2 * Math.PI * 5 * t) + 40 * (t / durS); // vibrato + slight rise
    phase += (2 * Math.PI * f) / SR;
    buf[s0 + i] += a * env * (Math.sin(phase) + 0.25 * Math.sin(2 * phase) + 0.08 * gauss());
  }
}

/** Speech-like chatter: harmonic stack with jumping F0, formant-ish weighting, syllabic AM. */
function addChatter(buf: Float32Array, startS: number, durS: number, db: number) {
  const a = dbToAmp(db);
  const s0 = Math.floor(startS * SR);
  const n = Math.floor(durS * SR);
  let f0 = 140;
  let formant = 1200;
  const phases = new Float64Array(30);
  for (let i = 0; i < n && s0 + i < buf.length; i++) {
    if (i % Math.floor(SR * 0.08) === 0) {
      f0 = 100 + rand() * 150;
      formant = 500 + rand() * 2500;
    }
    const t = i / SR;
    const syl = Math.max(0, Math.sin(2 * Math.PI * 4 * t)) ** 2;
    let v = 0;
    for (let h = 1; h <= 30; h++) {
      phases[h - 1] += (2 * Math.PI * f0 * h) / SR;
      const fh = f0 * h;
      v += Math.sin(phases[h - 1]) / (1 + ((fh - formant) / 400) ** 2) / h ** 0.3;
    }
    buf[s0 + i] += a * syl * (0.3 * v + 0.2 * gauss());
  }
}

/** Kettle coming to boil: broadband hiss band-limited ~2–6 kHz, swelling. */
function addKettle(buf: Float32Array, startS: number, durS: number, db: number) {
  const a = dbToAmp(db);
  const s0 = Math.floor(startS * SR);
  const n = Math.floor(durS * SR);
  let lp = 0;
  let hpPrev = 0;
  for (let i = 0; i < n && s0 + i < buf.length; i++) {
    const g = gauss();
    lp = 0.5 * lp + 0.5 * g; // crude low-pass
    const band = lp - hpPrev; // crude high-pass of the low-passed noise
    hpPrev = 0.9 * hpPrev + 0.1 * lp;
    buf[s0 + i] += a * (i / n) * band;
  }
}

/** Pan/ladle clang: short inharmonic ringing burst. */
function addClang(buf: Float32Array, startS: number, db: number) {
  const a = dbToAmp(db);
  const s0 = Math.floor(startS * SR);
  const partials = [1180, 2730, 4020];
  for (let i = 0; i < SR * 0.4 && s0 + i < buf.length; i++) {
    const t = i / SR;
    let v = 0;
    for (const p of partials) v += Math.sin(2 * Math.PI * p * t);
    buf[s0 + i] += a * Math.exp(-t / 0.08) * v;
  }
}

// ---- AnalyserNode emulation ----

const win = new Float64Array(FFT);
for (let i = 0; i < FFT; i++) {
  const x = (2 * Math.PI * i) / FFT;
  win[i] = 0.42 - 0.5 * Math.cos(x) + 0.08 * Math.cos(2 * x); // Blackman, alpha 0.16
}

function fftMag(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k + len / 2] = re[i + k] - ar;
        im[i + k + len / 2] = im[i + k] - ai;
        re[i + k] += ar;
        im[i + k] += ai;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

function runDetector(buf: Float32Array) {
  const core = new WhistleCore();
  const binHz = SR / FFT;
  const smoothed = new Float64Array(FFT / 2);
  const db = new Float32Array(FFT / 2);
  const re = new Float64Array(FFT);
  const im = new Float64Array(FFT);
  const hop = SR / FPS;
  const countedAt: number[] = [];
  let candidateFrames = 0;
  for (let end = FFT; end <= buf.length; end += hop) {
    for (let i = 0; i < FFT; i++) {
      re[i] = buf[end - FFT + i] * win[i];
      im[i] = 0;
    }
    fftMag(re, im);
    for (let k = 0; k < FFT / 2; k++) {
      const mag = Math.hypot(re[k], im[k]) / FFT;
      smoothed[k] = SMOOTHING * smoothed[k] + (1 - SMOOTHING) * mag;
      db[k] = 20 * Math.log10(smoothed[k] + 1e-12);
    }
    const nowMs = (end / SR) * 1000;
    const f = core.process(db, binHz, nowMs, thresholdDbFor(sensitivity));
    if (f.whistleFrame) candidateFrames++;
    if (f.started) countedAt.push(end / SR);
  }
  return { count: core.count, countedAt, candidateFrames };
}

// ---- scenarios ----

interface Scenario {
  name: string;
  seconds: number;
  expect: number;
  /** only meaningful at/above this sensitivity (quiet sources sit under the loudness floor) */
  minSensitivity?: number;
  build: (b: Float32Array) => void;
}

const scenarios: Scenario[] = [
  {
    name: '4 whistles, loud, room tone',
    seconds: 40,
    expect: 4,
    build: (b) => {
      addNoise(b, -55);
      [2, 11, 20, 30].forEach((t, i) => addWhistle(b, t, 2 + i * 0.4, 2400 + i * 350, -14));
    },
  },
  {
    name: '3 whistles, far away (-36 dB)',
    seconds: 30,
    expect: 3,
    minSensitivity: 0.8,
    build: (b) => {
      addNoise(b, -58);
      [2, 12, 22].forEach((t) => addWhistle(b, t, 1.8, 3100, -36));
    },
  },
  {
    name: '1 long 7 s whistle (no double count)',
    seconds: 12,
    expect: 1,
    build: (b) => {
      addNoise(b, -55);
      addWhistle(b, 2, 7, 2800, -16);
    },
  },
  {
    name: 'kitchen chatter only',
    seconds: 20,
    expect: 0,
    build: (b) => {
      addNoise(b, -55);
      addChatter(b, 1, 18, -20);
    },
  },
  {
    name: 'kettle boiling hiss',
    seconds: 15,
    expect: 0,
    build: (b) => {
      addNoise(b, -55);
      addKettle(b, 1, 13, -18);
    },
  },
  {
    name: 'pan clangs every 1.5 s',
    seconds: 15,
    expect: 0,
    build: (b) => {
      addNoise(b, -55);
      for (let t = 1; t < 14; t += 1.5) addClang(b, t, -12);
    },
  },
  {
    name: '2 whistles over chatter + kettle',
    seconds: 25,
    expect: 2,
    build: (b) => {
      addNoise(b, -55);
      addChatter(b, 0.5, 24, -24);
      addKettle(b, 0.5, 24, -26);
      addWhistle(b, 4, 2.2, 2600, -14);
      addWhistle(b, 15, 2.5, 3300, -14);
    },
  },
];

console.log(`seeti detector harness — sensitivity ${sensitivity} (floor ${thresholdDbFor(sensitivity)} dBFS)\n`);
let failures = 0;
for (const s of scenarios) {
  if (sensitivity < (s.minSensitivity ?? 0)) {
    console.log(`SKIP  ${s.name.padEnd(38)} (needs sensitivity >= ${s.minSensitivity})`);
    continue;
  }
  const buf = new Float32Array(Math.floor(s.seconds * SR));
  s.build(buf);
  const t0 = performance.now();
  const r = runDetector(buf);
  const ok = r.count === s.expect;
  if (!ok) failures++;
  const at = r.countedAt.map((t) => t.toFixed(1) + 's').join(' ') || '-';
  console.log(
    `${ok ? 'PASS' : 'FAIL'}  ${s.name.padEnd(38)} expected ${s.expect}  got ${r.count}  ` +
      `(counted at ${at}; ${r.candidateFrames} whistle-like frames; ${(performance.now() - t0).toFixed(0)} ms)`,
  );
}
console.log(`\n${failures} failing scenario(s) of ${scenarios.length}`);
process.exit(failures ? 1 : 0);
