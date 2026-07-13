function mergeChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function encodeWav(chunks, sampleRate) {
  const samples = mergeChunks(chunks);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  samples.forEach((sample) => {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  });
  return new Blob([buffer], { type: "audio/wav" });
}

export class MicrophoneCapture {
  static async open() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone capture is not supported by this browser.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    return new MicrophoneCapture(stream);
  }

  constructor(stream) {
    this.stream = stream;
    this.context = new AudioContext();
    this.source = this.context.createMediaStreamSource(stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
    this.mode = "idle";
    this.enabled = true;
    this.chunks = [];
    this.preRoll = [];
    this.silenceSamples = 0;
    this.noiseFloor = 0.008;
    this.onUtterance = null;
    this.processor.onaudioprocess = (event) => this.handleAudio(event);
  }

  async resume() {
    if (this.context.state === "suspended") await this.context.resume();
  }

  startManual() {
    this.mode = "manual";
    this.chunks = [];
    this.enabled = true;
  }

  stopManual() {
    if (this.mode !== "manual") return null;
    const chunks = this.chunks;
    this.mode = "idle";
    this.chunks = [];
    return chunks.length ? encodeWav(chunks, this.context.sampleRate) : null;
  }

  startVoiceActivity(onUtterance) {
    this.mode = "vad";
    this.enabled = true;
    this.onUtterance = onUtterance;
    this.resetUtterance();
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    if (!enabled) this.resetUtterance();
  }

  resetUtterance() {
    this.chunks = [];
    this.preRoll = [];
    this.silenceSamples = 0;
  }

  handleAudio(event) {
    if (!this.enabled || this.mode === "idle") return;
    const input = event.inputBuffer.getChannelData(0);
    const chunk = new Float32Array(input);
    if (this.mode === "manual") {
      this.chunks.push(chunk);
      return;
    }

    let sum = 0;
    for (let i = 0; i < chunk.length; i += 1) sum += chunk[i] * chunk[i];
    const rms = Math.sqrt(sum / chunk.length);
    const threshold = Math.max(0.015, this.noiseFloor * 3);
    const speaking = rms > threshold;

    if (!this.chunks.length) {
      this.noiseFloor = this.noiseFloor * 0.97 + Math.min(rms, 0.03) * 0.03;
      this.preRoll.push(chunk);
      const maxPreRoll = Math.ceil((this.context.sampleRate * 0.3) / chunk.length);
      if (this.preRoll.length > maxPreRoll) this.preRoll.shift();
      if (speaking) {
        this.chunks = this.preRoll;
        this.preRoll = [];
      }
      return;
    }

    this.chunks.push(chunk);
    this.silenceSamples = speaking ? 0 : this.silenceSamples + chunk.length;
    const totalSamples = this.chunks.reduce((total, item) => total + item.length, 0);
    const duration = totalSamples / this.context.sampleRate;
    const silenceDuration = this.silenceSamples / this.context.sampleRate;

    if ((duration >= 0.45 && silenceDuration >= 0.7) || duration >= 20) {
      const blob = encodeWav(this.chunks, this.context.sampleRate);
      this.resetUtterance();
      this.onUtterance?.(blob);
    }
  }

  close() {
    this.mode = "idle";
    this.processor.disconnect();
    this.source.disconnect();
    this.stream.getTracks().forEach((track) => track.stop());
    this.context.close();
  }
}
