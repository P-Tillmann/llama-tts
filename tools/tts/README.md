# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```

## Piper (VITS)

Piper voices run from the mmproj alone, there is no backbone model (`-m` is not needed).

Available params:
- `--tts-phonemizer` points to an external text-to-phoneme program, [tts-phonemizer-espeak](https://github.com/P-Tillmann/tts-phonemizer-espeak) (GPL-3.0, a separate download; llama.cpp does not link espeak-ng). With it, `-p` is text
- without `--tts-phonemizer`, `-p` is a phoneme string (IPA, as produced by espeak-ng for the voice), one utterance per line
- `--tts-speaker-id` selects the speaker of a multi-speaker voice (default: 0)
- `--tts-speed` changes the speaking rate, > 1 is faster (default: from the voice config)
- `--seed` fixes the noise of the duration predictor and the prior

Example usage:

```sh
llama-tts -mm piper-en_US-lessac-medium.gguf \
    --tts-phonemizer tts-phonemizer-espeak \
    -p "Hello world. Is this thing on?" \
    --output out.wav

# phonemes instead of text
llama-tts -mm piper-en_US-lessac-medium.gguf \
    -p "həlˈoʊ wˈɜːld." \
    --output out.wav
```

**Note for GGUF conversion:**

A voice is an `.onnx` file plus its `.onnx.json` config from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices). The license is per voice, see its `MODEL_CARD`. The converter is not part of `convert_hf_to_gguf.py` yet.

## Kokoro

Kokoro-82M runs from the mmproj alone, like Piper. The voice packs are stored in the GGUF; `--tts-speaker-id` selects one (0 = first voice given to the converter).

Available params:
- `--tts-phonemizer` points to [tts-phonemizer-espeak](https://github.com/P-Tillmann/tts-phonemizer-espeak); with it `-p` is text. American English uses a port of the misaki G2P with its lexicon (stored in the GGUF); espeak-ng is only used for unknown words. Other languages go through espeak-ng and the misaki mapping. Heteronyms (read, present, ...) are chosen with simple context rules, there is no part-of-speech tagger
- without `--tts-phonemizer`, `-p` is a Kokoro phoneme string (misaki format), one paragraph per line
- the language follows the first letter of the voice name: `a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p` Brazilian Portuguese (Japanese and Chinese are not supported)
- `--tts-speed` changes the speaking rate (default 1)
- long input is split into chunks of at most 510 phonemes, at sentence ends when possible

Example usage:

```sh
llama-tts -mm kokoro-v1_0.gguf \
    --tts-phonemizer tts-phonemizer-espeak \
    --tts-speaker-id 0 \
    -p "Hello world. Is this thing on?" \
    --output out.wav
```

**Note for GGUF conversion:**

The converter is not part of `convert_hf_to_gguf.py` yet. It takes `kokoro-v1_0.pth`, `config.json` and one or more voice packs (`voices/*.pt`) from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), and the US English lexicons of [misaki](https://github.com/hexgrad/misaki) (all Apache-2.0).
