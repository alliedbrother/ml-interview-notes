---
order: 11
description: Audio representations and spectrograms, the ASR arc from HMM-GMM to CTC to Whisper, streaming and RNN-T, TTS and neural vocoders, speaker tasks, audio LLMs, and WER done properly.
meta: NLP · speech
---

# Speech and Audio

Speech is text plus everything text throws away: timing, prosody, speaker
identity, emotion, and acoustic environment. That extra information is why speech
systems need their own representations and architectures, and why the tasks fail
in ways text systems do not.

## Audio representations

Raw audio is a waveform: amplitude sampled at 16 kHz for speech, 44.1 kHz for
music. One second of 16 kHz audio is 16,000 numbers, which is far too long a
sequence to model directly with attention.

```mermaid
flowchart LR
    W["waveform<br/>16000 samples per second"] --> F["frame:<br/>25 ms windows,<br/>10 ms hop"]
    F --> H["window function<br/>Hann, to reduce spectral leakage"]
    H --> FFT["FFT per frame"]
    FFT --> M["mel filterbank<br/>warps frequency to match<br/>human perception"]
    M --> L["log scale<br/>matches loudness perception"]
    L --> S["log-mel spectrogram<br/>80 bins x 100 frames per second"]
```

**The framing rate is the key number**: 25 ms windows with a 10 ms hop gives 100
frames per second, so one second becomes 100 timesteps of 80 features instead of
16,000 samples — a 160× reduction, and now a tractable sequence length.

| Representation | Note |
|---|---|
| Raw waveform | maximum information; used by WaveNet, SEW, some end-to-end models |
| Spectrogram (STFT) | time–frequency magnitude |
| **Log-mel spectrogram** | mel-warped and log-scaled; the standard input |
| MFCC | DCT of log-mel; decorrelated, compact — the classical feature, now largely superseded |
| **Learned SSL features** | wav2vec 2.0, HuBERT, WavLM — self-supervised, and better than hand-designed features |
| Discrete audio tokens | EnCodec, SoundStream — enable audio LLMs |

**The mel scale** approximates human pitch perception: we distinguish 200 Hz from
300 Hz easily and 8,000 Hz from 8,100 Hz not at all. Warping the frequency axis
accordingly concentrates resolution where perception is sharp.

**Discrete audio tokens are the enabling technology for audio LLMs.** A neural
codec compresses audio into a sequence of integers, which a transformer can model
with exactly the same machinery it uses for text — turning speech generation into
next-token prediction.

## Automatic speech recognition

### The arc

| Era | Approach |
|---|---|
| 1980s–2000s | **HMM-GMM**: HMM for temporal structure, GMM for acoustics, separate pronunciation lexicon and n-gram language model |
| 2010s | HMM-DNN: replace the GMM with a neural network; large accuracy gain |
| 2015+ | **CTC**: end-to-end, no alignment needed, no lexicon |
| 2016+ | **RNN-T**: CTC plus a prediction network; the streaming standard |
| 2017+ | Attention encoder–decoder (LAS): full seq2seq |
| 2020+ | **Self-supervised pretraining** (wav2vec 2.0, HuBERT) then fine-tune |
| 2022+ | **Whisper**: weakly supervised at massive scale, multilingual, multitask |

### CTC

The alignment problem: an input of $T$ audio frames maps to $U \ll T$ output
characters, with no per-frame labels.

**Connectionist temporal classification** introduces a blank symbol and defines
the probability of a label sequence as the sum over **all** alignments that
collapse to it:

$$p(\mathbf{y}\mid\mathbf{x}) = \sum_{\pi\in\mathcal{B}^{-1}(\mathbf{y})}\prod_{t=1}^{T}p(\pi_t\mid\mathbf{x})$$

The collapse rule removes repeated symbols and then blanks, so `h-e-l-l-o` needs
a blank between the two `l`s to survive: `hel_lo`. That is precisely why the
blank exists.

The sum has exponentially many terms and is computed in $O(TU)$ by a
forward–backward dynamic program, which is what makes the loss differentiable and
trainable.

**CTC's limitation is its conditional independence assumption**: outputs are
independent given the input, so the model has no internal language model. That is
why CTC systems are almost always paired with an external LM at decoding time
(beam search with shallow fusion).

**RNN-Transducer** removes that assumption by adding a prediction network
conditioned on previous outputs, combining acoustic and language modelling in one
architecture. It is naturally streaming, and it is what most production
on-device ASR uses.

### Whisper

Trained on 680,000 hours of weakly supervised multilingual audio scraped from the
web, as a single encoder–decoder model handling transcription, translation,
language identification, and timestamps through **special tokens in the decoder
prompt**.

| Strength | Weakness |
|---|---|
| Robust across accents, noise, and domains | not streaming — processes 30-second windows |
| 99 languages in one model | **hallucinates** on silence and non-speech audio |
| No fine-tuning needed for most uses | repetition loops on long or unusual audio |
| Multitask through prompt tokens | no speaker diarisation |
| Open weights | high latency for real-time use |

**Whisper's hallucination on silence is a genuine production problem**: given
non-speech input it can emit fluent transcript text (frequently subtitle-corpus
artefacts like "Thank you for watching"). Mitigate with voice-activity detection
before transcription, a no-speech probability threshold, and repetition
detection.

Faster-whisper (CTranslate2) and WhisperX (with forced alignment and diarisation)
are the practical deployment paths.

### Streaming

Real-time ASR must emit output before the utterance ends, which changes the
architecture.

| Requirement | Mechanism |
|---|---|
| Causal or limited-lookahead encoding | chunked or causal attention |
| Incremental decoding | RNN-T, or a streaming transformer transducer |
| Endpointing | detect when the speaker has finished |
| Partial hypotheses | show provisional text, revise as more audio arrives |

The trade is direct: more lookahead gives better accuracy and higher latency.
Production systems tune this per use case — dictation tolerates more latency than
a voice assistant.

## Text to speech

```mermaid
flowchart LR
    T["text"] --> N["text normalisation<br/>numbers, dates, abbreviations"]
    N --> G["grapheme to phoneme<br/>optional in end-to-end systems"]
    G --> A["acoustic model<br/>text to mel spectrogram"]
    A --> V["vocoder<br/>mel spectrogram to waveform"]
    V --> W["audio"]
```

| Stage | Models |
|---|---|
| Acoustic | Tacotron 2 (autoregressive), FastSpeech 2 (non-autoregressive, controllable duration), VITS (end-to-end with a VAE and flows) |
| **Vocoder** | WaveNet (excellent, very slow), WaveGlow, **HiFi-GAN** (fast and high quality), BigVGAN |
| End-to-end | VITS, StyleTTS 2, VALL-E, XTTS |

**Text normalisation is the unglamorous stage that breaks systems.** "Dr. Smith
lives at 123 Dr." — the first is "Doctor", the second is "Drive". "1/2" is "one
half" or "January second". "$1.5M" is "one point five million dollars".
Rule-based normalisers handle most cases; neural normalisers handle more and fail
less predictably.

**Modern TTS is essentially solved for quality** and the interesting problems
are elsewhere: **voice cloning** from a few seconds of reference audio, **prosody
and emotion control**, **latency** for conversational agents, and **streaming**
synthesis that starts speaking before the full text is generated.

That last one matters for LLM voice interfaces: the pipeline is ASR → LLM → TTS,
and each stage adds latency. Streaming all three — partial transcripts feeding a
streaming LLM feeding a streaming vocoder — is what makes a voice assistant feel
responsive rather than sluggish.

## Speaker and audio tasks

| Task | Description |
|---|---|
| **Speaker identification** | who is speaking, from a known set |
| **Speaker verification** | is this the claimed speaker? (biometric) |
| **Diarisation** | "who spoke when" — segmentation plus clustering |
| Voice activity detection | speech versus non-speech |
| Language identification | which language |
| **Source separation** | isolate voices or instruments from a mixture |
| Keyword spotting | wake words, on-device and low power |
| Emotion recognition | affect from prosody |
| Audio classification | events, scenes, music genre |
| Music transcription | audio to notation |

**Speaker embeddings** (x-vectors, ECAPA-TDNN) are the shared substrate: a
fixed-length vector per utterance where cosine distance measures speaker
similarity. Verification, diarisation, and clustering all reduce to comparing
these.

**Diarisation combined with ASR** is what most real transcription products
actually need — a meeting transcript with speaker labels — and it remains harder
than either component alone, particularly with overlapping speech.

## Self-supervised speech models

| Model | Objective |
|---|---|
| **wav2vec 2.0** | contrastive prediction of quantised latent speech units from masked context |
| **HuBERT** | masked prediction of cluster IDs from an offline $k$-means over features, refined iteratively |
| WavLM | HuBERT plus simulated overlapped speech and denoising — better for speaker tasks |
| Whisper encoder | weakly supervised, but its features transfer well |
| EnCodec / SoundStream | neural codecs producing discrete tokens |

**The impact is on data requirements.** wav2vec 2.0 fine-tuned on **10 minutes**
of labelled speech reaches word error rates that previously needed hundreds of
hours. For low-resource languages — most of the world's languages — this is the
difference between a possible and an impossible ASR system.

## Audio language models

The convergence: tokenise audio with a neural codec, then model the tokens with a
transformer exactly as with text.

| System | Capability |
|---|---|
| AudioLM | continue speech or music from a prompt |
| VALL-E | zero-shot voice cloning from a 3-second sample |
| MusicGen / MusicLM | text-to-music |
| AudioGen | text-to-sound-effect |
| Speech-in LLMs (Qwen-Audio, Gemini, GPT-4o) | audio understanding as one modality among several |
| Speech-to-speech | skip the text bottleneck entirely; preserve prosody and emotion |

**Speech-to-speech is the architecturally interesting direction.** The
conventional pipeline (ASR → LLM → TTS) discards prosody, emotion, and speaker
characteristics at the first stage and cannot recover them at the last. A model
operating on audio tokens throughout preserves them, and can also respond much
faster because it does not wait for a full transcript.

## Evaluation

### Word error rate

$$\mathrm{WER} = \frac{S + D + I}{N}$$

Substitutions, deletions, and insertions divided by reference words, computed by
the **edit-distance dynamic program**. It can exceed 100% when insertions
dominate.

| Caveat | Detail |
|---|---|
| **Normalisation dominates** | casing, punctuation, numbers ("5" vs "five"), contractions — a WER comparison without identical normalisation is meaningless |
| Not all errors are equal | a wrong digit in an account number matters more than "a" vs "the" |
| Morphologically rich languages | word-level WER is harsh; use **CER** instead |
| Speaker-attributed WER | for diarised multi-speaker transcripts |

**Always publish the normalisation.** Whisper ships a text normaliser precisely
because WER numbers are otherwise incomparable, and most disputes about ASR
quality turn out to be disputes about normalisation.

| Task | Metric |
|---|---|
| ASR | WER, CER |
| ASR (downstream) | task success, entity error rate |
| Diarisation | **DER** — diarisation error rate (missed, false alarm, confusion) |
| Speaker verification | **EER** — equal error rate; ROC-AUC |
| TTS quality | **MOS** (human 1–5), UTMOS (predicted MOS) |
| TTS intelligibility | WER of an ASR system on the synthesised audio |
| Voice similarity | speaker-embedding cosine similarity |
| Source separation | SDR, SI-SNR |

**Using ASR-WER to evaluate TTS** is the neat trick worth knowing: synthesise
text, transcribe it, and measure the error rate. It gives an automatic,
reproducible intelligibility number without human raters.

## Production concerns

| Concern | Handling |
|---|---|
| Audio quality | 16 kHz mono is standard for speech; check for clipping and DC offset |
| Noise and reverberation | augment training with noise and room impulse responses |
| Accents and dialects | a well-documented equity gap; measure per-group WER explicitly |
| Code-switching | multilingual models; mid-utterance language changes are hard |
| Domain vocabulary | biasing, hotwords, or a custom LM for names and jargon |
| Long audio | chunk with overlap; align and stitch |
| Latency | streaming models; measure time-to-first-word |
| Cost | on-device for wake words and simple commands; server for full ASR |
| Privacy | on-device processing; audio is biometric data under several regulations |

**Per-group WER measurement is not optional.** ASR error rates differ
substantially by accent, dialect, age, and gender, and aggregate WER hides it.
Published audits have found error rates roughly twice as high for some speaker
groups as for others. This is both a quality problem and a fairness problem, and
you cannot fix what you do not measure.

## Self-check

1. Why convert a waveform to a log-mel spectrogram? Give the sequence-length
   arithmetic.
2. What problem does CTC solve, and what makes its loss tractable?
3. Why does CTC need a blank symbol? Give a word that demonstrates it.
4. What does RNN-T add over CTC, and why does that matter for streaming?
5. Why does Whisper hallucinate on silence, and what are two mitigations?
6. Why is a WER comparison meaningless without stated normalisation?
7. How would you evaluate TTS intelligibility automatically?

## Where to go next

- [Text Preprocessing](./text-preprocessing.md) — normalisation, which decides
  WER.
- [Language Models](./language-models.md) — the LM half of a speech pipeline.
- [NLP Evaluation](./nlp-evaluation.md) — metrics across generation tasks.
