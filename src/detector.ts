// Pressure-cooker whistle detector built on the Web Audio API.
//
// A cooker whistle is a loud, *narrowband*, frequency-stable tone (~700–4500 Hz)
// sustained for half a second or more. Speech is broadband, heavily modulated,
// and its dominant frequency jumps around every few tens of milliseconds.
// So a frame only counts as "whistle-like" when ALL of these hold:
//   1. the loudest bin in the band is above an absolute loudness threshold
//   2. that peak stands far above the band's median level (narrowband tonality)
//   3. the peak frequency has stayed put over the recent past (stability)
// and the whistle state only latches after enough consecutive whistle-like
// frames (sustain), with a refractory period between counts.

export interface DetectorState {
  running: boolean;
  whistling: boolean;
  count: number;
  peakFreq: number; // Hz, 0 when nothing prominent
  peakDb: number; // dBFS of the dominant peak
  prominenceDb: number; // peak vs band median
  holdMs: number; // how long current candidate tone has been held
}

export interface DetectorOptions {
  /** 0..1, higher = more sensitive (lower loudness threshold) */
  sensitivity: number;
  onWhistleStart: () => void;
  onWhistleEnd: (count: number) => void;
  onFrame: (state: DetectorState) => void;
}

const BAND_LOW_HZ = 700;
const BAND_HIGH_HZ = 4500;
const PROMINENCE_DB = 17; // peak must beat band median by this much
const STABILITY_HZ = 160; // max drift from rolling mean peak freq
const SUSTAIN_MS = 450; // tone must hold this long before it counts
const RELEASE_MS = 350; // tone must vanish this long before whistle ends
const REFRACTORY_MS = 3000; // min gap between two counted whistles

export class WhistleDetector {
  private opts: DetectorOptions;
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  private freqData: Float32Array<ArrayBuffer> = new Float32Array(0);
  private raf = 0;

  private whistling = false;
  private count = 0;
  private holdStart = 0; // ts when candidate tone first appeared
  private lastGoodFrame = 0; // ts of last whistle-like frame
  private lastCountAt = -Infinity;
  private recentFreqs: number[] = [];

  constructor(opts: DetectorOptions) {
    this.opts = opts;
  }

  setSensitivity(s: number) {
    this.opts.sensitivity = s;
  }

  /** dBFS loudness floor: sensitivity 0 → -28 dB (needs to be very loud), 1 → -58 dB */
  private get thresholdDb() {
    return -28 - 30 * this.opts.sensitivity;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Browser DSP is tuned for voice and will suppress/duck a sustained
        // tone — turn all of it off so the whistle reaches us untouched.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    this.ctx = new AudioContext();
    await this.ctx.resume();
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 8192;
    this.analyser.smoothingTimeConstant = 0.4;
    src.connect(this.analyser);
    this.freqData = new Float32Array(this.analyser.frequencyBinCount);
    this.loop();
  }

  stop() {
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.ctx = null;
    this.stream = null;
    this.analyser = null;
    this.whistling = false;
    this.recentFreqs = [];
    this.emit(false, 0, -120, 0, 0);
  }

  resetCount() {
    this.count = 0;
  }

  private emit(
    whistling: boolean,
    peakFreq: number,
    peakDb: number,
    prominenceDb: number,
    holdMs: number,
  ) {
    this.opts.onFrame({
      running: !!this.ctx,
      whistling,
      count: this.count,
      peakFreq,
      peakDb,
      prominenceDb,
      holdMs,
    });
  }

  private loop = () => {
    const analyser = this.analyser;
    const ctx = this.ctx;
    if (!analyser || !ctx) return;
    this.raf = requestAnimationFrame(this.loop);

    analyser.getFloatFrequencyData(this.freqData);
    const binHz = ctx.sampleRate / analyser.fftSize;
    const lo = Math.floor(BAND_LOW_HZ / binHz);
    const hi = Math.min(Math.ceil(BAND_HIGH_HZ / binHz), this.freqData.length - 1);

    let peakBin = lo;
    let peakDb = -Infinity;
    for (let i = lo; i <= hi; i++) {
      if (this.freqData[i] > peakDb) {
        peakDb = this.freqData[i];
        peakBin = i;
      }
    }
    const band = Array.from(this.freqData.subarray(lo, hi + 1)).sort((a, b) => a - b);
    const medianDb = band[band.length >> 1];
    const prominence = peakDb - medianDb;
    const peakFreq = peakBin * binHz;

    const now = performance.now();
    const loudEnough = peakDb > this.thresholdDb;
    const narrowband = prominence > PROMINENCE_DB;

    // Frequency stability vs the rolling mean of recent peaks
    this.recentFreqs.push(peakFreq);
    if (this.recentFreqs.length > 12) this.recentFreqs.shift();
    const mean = this.recentFreqs.reduce((a, b) => a + b, 0) / this.recentFreqs.length;
    const stable = this.recentFreqs.length >= 6 && Math.abs(peakFreq - mean) < STABILITY_HZ;

    const whistleFrame = loudEnough && narrowband && stable;

    if (whistleFrame) {
      if (this.holdStart === 0) this.holdStart = now;
      this.lastGoodFrame = now;
    } else if (now - this.lastGoodFrame > RELEASE_MS) {
      this.holdStart = 0;
      if (this.whistling) {
        this.whistling = false;
        this.opts.onWhistleEnd(this.count);
      }
    }

    const holdMs = this.holdStart ? now - this.holdStart : 0;

    if (
      !this.whistling &&
      this.holdStart !== 0 &&
      holdMs >= SUSTAIN_MS &&
      now - this.lastCountAt >= REFRACTORY_MS
    ) {
      this.whistling = true;
      this.count += 1;
      this.lastCountAt = now;
      this.opts.onWhistleStart();
    }

    this.emit(this.whistling, whistleFrame ? peakFreq : 0, peakDb, prominence, holdMs);
  };
}
