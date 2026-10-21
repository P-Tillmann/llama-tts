#!/usr/bin/env python3
# Convert Kokoro-82M (kokoro-v1_0.pth + config.json + voice packs) to a GGUF for llama-tts (--mmproj).
# Needs: gguf (from gguf-py in this repo), numpy, plus onnx (Piper) or torch (Kokoro).
#
# - weight norm is folded: W = g * v / ||v||, norm over all dims but 0 (IC for ConvTranspose1d)
# - AdaIN / AdaLN fc is split into gamma (with +1 folded into the bias) and beta
# - LSTM: both directions stacked, w_ih [8H, IN], bias b_ih + b_hh [8H], w_hh [2, 4H, H]; gate order i, f, g, o
# - token_type_embeddings[0] is folded into the position embeddings, the ALBERT pooler is dropped
# - Snake alpha and 1/alpha are stored as [1, C, 1] (ggml ne [1, C]) for time-major broadcasting
# - depthwise ConvTranspose1d (AdainResBlk1d pool) taps are stored as [K, C]
# - voice packs [510, 1, 256] are stacked into a.gen.voices [n_voices, 510, 256], one speaker per voice

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import gguf

KEY = "clip.gen.audio."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="kokoro-v1_0.pth")
    ap.add_argument("--config", required=True, help="config.json")
    ap.add_argument("--voice", action="append", default=[], help="voice pack .pt, repeat for more voices")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--lexicon", nargs=2, metavar=("GOLD", "SILVER"),
                    help="misaki us_gold.json and us_silver.json (Apache-2.0, https://github.com/hexgrad/misaki), for the English G2P")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ist = cfg["istftnet"]
    expect = {"upsample_rates": [10, 6], "upsample_kernel_sizes": [20, 12], "gen_istft_n_fft": 20, "gen_istft_hop_size": 5,
              "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5]] * 3, "upsample_initial_channel": 512}
    for k, v in expect.items():
        if ist[k] != v:
            sys.exit(f"istftnet.{k} = {ist[k]}, only {v} is supported")
    if cfg["hidden_dim"] != 512 or cfg["style_dim"] != 128 or cfg["n_layer"] != 3 or cfg["max_dur"] != 50 or cfg["text_encoder_kernel_size"] != 5:
        sys.exit("unsupported kokoro config")

    raw = torch.load(args.model, map_location="cpu", weights_only=True)
    sd = {}
    for top, d in raw.items():
        for k, v in d.items():
            sd[f"{top}.{k.removeprefix('module.')}"] = v.float().numpy()
    used = set()

    def get(k):
        used.add(k)
        return sd[k]

    def wn(p):
        # weight norm, dim 0
        if p + ".weight" in sd:
            return get(p + ".weight")
        g, v = get(p + ".weight_g"), get(p + ".weight_v")
        n = np.sqrt((v.astype(np.float64) ** 2).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
        return (g * v / n).astype(np.float32)

    tensors = {}

    def put(name, a):
        assert name not in tensors, name
        tensors[name] = np.ascontiguousarray(a, dtype=np.float32)

    def put_conv(dst, src, bias=True):
        put(dst + ".weight", wn(src))
        if bias:
            put(dst + ".bias", get(src + ".bias"))

    def put_linear(dst, src):
        put(dst + ".weight", get(src + ".weight"))
        put(dst + ".bias", get(src + ".bias"))

    def put_ada(dst, src):
        # fc: style -> [gamma; beta]
        w, b = get(src + ".fc.weight"), get(src + ".fc.bias")
        c = w.shape[0] // 2
        put(dst + ".gamma.weight", w[:c])
        put(dst + ".gamma.bias", b[:c] + 1.0)
        put(dst + ".beta.weight", w[c:])
        put(dst + ".beta.bias", b[c:])

    def put_lstm(dst, src):
        w_ih = np.concatenate([get(src + ".weight_ih_l0"), get(src + ".weight_ih_l0_reverse")], 0)
        b = np.concatenate([get(src + ".bias_ih_l0") + get(src + ".bias_hh_l0"),
                            get(src + ".bias_ih_l0_reverse") + get(src + ".bias_hh_l0_reverse")], 0)
        w_hh = np.stack([get(src + ".weight_hh_l0"), get(src + ".weight_hh_l0_reverse")], 0)
        put(dst + ".w_ih", w_ih)
        put(dst + ".b", b)
        put(dst + ".w_hh", w_hh)

    def put_adain_resblk(dst, src):
        # AdainResBlk1d (StyleTTS2)
        put_conv(dst + ".conv1", src + ".conv1")
        put_conv(dst + ".conv2", src + ".conv2")
        put_ada(dst + ".norm1", src + ".norm1")
        put_ada(dst + ".norm2", src + ".norm2")
        if src + ".conv1x1.weight_v" in sd:
            put_conv(dst + ".conv1x1", src + ".conv1x1", bias=False)
        if src + ".pool.weight_v" in sd:
            w = wn(src + ".pool")  # [C, 1, 3]
            put(dst + ".pool.weight", w[:, 0, :].T)
            put(dst + ".pool.bias", get(src + ".pool.bias"))

    def put_adain_resblock1(dst, src):
        # AdaINResBlock1 (iSTFTNet generator), 3 dilations
        for d in range(3):
            put_conv(f"{dst}.c1.{d}", f"{src}.convs1.{d}")
            put_conv(f"{dst}.c2.{d}", f"{src}.convs2.{d}")
            put_ada(f"{dst}.n1.{d}", f"{src}.adain1.{d}")
            put_ada(f"{dst}.n2.{d}", f"{src}.adain2.{d}")
            for a in (1, 2):
                alpha = get(f"{src}.alpha{a}.{d}")  # [1, C, 1]
                put(f"{dst}.alpha{a}.{d}", alpha)
                put(f"{dst}.ialpha{a}.{d}", 1.0 / alpha)

    # plbert (ALBERT, one shared layer)
    b = "bert."
    put("a.gen.bert.token_embd.weight", get(b + "embeddings.word_embeddings.weight"))
    put("a.gen.bert.position_embd.weight", get(b + "embeddings.position_embeddings.weight") + get(b + "embeddings.token_type_embeddings.weight")[0])
    put("a.gen.bert.embd_norm.weight", get(b + "embeddings.LayerNorm.weight"))
    put("a.gen.bert.embd_norm.bias", get(b + "embeddings.LayerNorm.bias"))
    put_linear("a.gen.bert.embd_proj", b + "encoder.embedding_hidden_mapping_in")
    lay = b + "encoder.albert_layer_groups.0.albert_layers.0."
    for src, dst in (("attention.query", "attn_q"), ("attention.key", "attn_k"), ("attention.value", "attn_v"), ("attention.dense", "attn_out"),
                     ("attention.LayerNorm", "ln1"), ("ffn", "ffn_up"), ("ffn_output", "ffn_down"), ("full_layer_layer_norm", "ln2")):
        put_linear(f"a.gen.bert.blk.0.{dst}", lay + src)
    for k in sd:
        if k.startswith(b + "pooler"):
            used.add(k)
    put_linear("a.gen.bert_proj", "bert_encoder")

    # prosody predictor, text side
    p = "predictor."
    for i in range(3):
        put_lstm(f"a.gen.dur_enc.{i}.lstm", f"{p}text_encoder.lstms.{2 * i}")
        put_ada(f"a.gen.dur_enc.{i}.norm", f"{p}text_encoder.lstms.{2 * i + 1}")
    put_lstm("a.gen.dur_lstm", p + "lstm")
    put_linear("a.gen.dur_proj", p + "duration_proj.linear_layer")

    # text encoder
    t = "text_encoder."
    put("a.gen.text_enc.token_embd.weight", get(t + "embedding.weight"))
    for i in range(3):
        put_conv(f"a.gen.text_enc.cnn.{i}.conv", f"{t}cnn.{i}.0")
        put(f"a.gen.text_enc.cnn.{i}.norm.weight", get(f"{t}cnn.{i}.1.gamma"))
        put(f"a.gen.text_enc.cnn.{i}.norm.bias", get(f"{t}cnn.{i}.1.beta"))
    put_lstm("a.gen.text_enc.lstm", t + "lstm")

    # prosody predictor, frame side
    put_lstm("a.gen.shared_lstm", p + "shared")
    for br in ("F0", "N"):
        for i in range(3):
            put_adain_resblk(f"a.gen.{br.lower()}.{i}", f"{p}{br}.{i}")
        put_conv(f"a.gen.{br.lower()}_proj", f"{p}{br}_proj")

    # decoder
    d = "decoder."
    put_conv("a.gen.dec.f0_conv", d + "F0_conv")
    put_conv("a.gen.dec.n_conv", d + "N_conv")
    put_conv("a.gen.dec.asr_res", d + "asr_res.0")
    put_adain_resblk("a.gen.dec.encode", d + "encode")
    for i in range(4):
        put_adain_resblk(f"a.gen.dec.decode.{i}", f"{d}decode.{i}")

    # generator
    g = d + "generator."
    put_linear("a.gen.gen.l_linear", g + "m_source.l_linear")
    for i in range(2):
        put_conv(f"a.gen.gen.noise_conv.{i}", f"{g}noise_convs.{i}")
        put_adain_resblock1(f"a.gen.gen.noise_res.{i}", f"{g}noise_res.{i}")
        put_conv(f"a.gen.gen.up.{i}", f"{g}ups.{i}")
    for j in range(6):
        put_adain_resblock1(f"a.gen.gen.res.{j}", f"{g}resblocks.{j}")
    put_conv("a.gen.gen.conv_post", g + "conv_post")

    unused = sorted(set(sd) - used)
    if unused:
        sys.exit(f"unused tensors: {unused[:10]} ...")

    # voices
    names = []
    packs = []
    for v in args.voice:
        pack = torch.load(v, map_location="cpu", weights_only=True).float().numpy()
        if pack.shape != (510, 1, 256):
            sys.exit(f"{v}: voice pack shape {pack.shape}")
        packs.append(pack[:, 0, :])
        names.append(Path(v).stem)
    if not packs:
        sys.exit("at least one --voice is needed")
    put("a.gen.voices", np.stack(packs, 0))

    # lexicon: gold entries first, then silver; heteronyms as "word\tTAG", None as ""
    lex_words, lex_phonemes = [], []
    gold   = json.loads(Path(args.lexicon[0]).read_text(encoding="utf-8")) if args.lexicon else {}
    silver = json.loads(Path(args.lexicon[1]).read_text(encoding="utf-8")) if args.lexicon else {}
    for w, v in gold.items():
        if isinstance(v, dict):
            lex_words.append(w)
            lex_phonemes.append(v["DEFAULT"] or "")
            for tag, ps in v.items():
                if tag != "DEFAULT":
                    lex_words.append(f"{w}\t{tag}")
                    lex_phonemes.append(ps or "")
        else:
            lex_words.append(w)
            lex_phonemes.append(v)
    n_gold = len(lex_words)
    for w, v in silver.items():
        lex_words.append(w)
        lex_phonemes.append(v)

    symbols = [""] * cfg["n_token"]
    for ph, i in cfg["vocab"].items():
        symbols[i] = ph

    gw = gguf.GGUFWriter(args.outfile, "clip")
    gw.add_name("kokoro-82m")
    gw.add_file_type(gguf.LlamaFileType.ALL_F32)
    gw.add_description("Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0)")
    gw.add_bool("clip.has_vision_encoder", False)
    gw.add_bool("clip.has_audio_encoder", False)
    gw.add_bool("clip.has_gen_audio_encoder", True)
    gw.add_string(KEY + "projector_type", "kokoro")
    gw.add_string(KEY + "model_variant", "v1.0")
    gw.add_uint32(KEY + "embedding_length", cfg["plbert"]["hidden_size"])
    gw.add_uint32(KEY + "feed_forward_length", cfg["plbert"]["intermediate_size"])
    gw.add_uint32(KEY + "block_count", cfg["plbert"]["num_hidden_layers"])
    gw.add_uint32(KEY + "projection_dim", cfg["hidden_dim"])
    gw.add_uint32(KEY + "attention.head_count", cfg["plbert"]["num_attention_heads"])
    gw.add_float32(KEY + "attention.layer_norm_epsilon", 1e-12)
    gw.add_uint32(KEY + "sample_rate", 24000)
    gw.add_float32(KEY + "noise_scale", 1.0)
    gw.add_float32(KEY + "length_scale", 1.0)
    gw.add_array(KEY + "symbols", symbols)
    gw.add_array(KEY + "speakers", names)
    if lex_words:
        gw.add_string(KEY + "lexicon_lang", "en-us")
        gw.add_uint32(KEY + "lexicon_n_gold", n_gold)
        gw.add_array(KEY + "lexicon_words", lex_words)
        gw.add_array(KEY + "lexicon_phonemes", lex_phonemes)
    for n, a in tensors.items():
        gw.add_tensor(n, a)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"{args.outfile}: {len(tensors)} tensors, {sum(a.size for a in tensors.values()) / 1e6:.2f}M params, voices {names}, "
          f"lexicon {n_gold} gold + {len(lex_words) - n_gold} silver")


if __name__ == "__main__":
    main()
