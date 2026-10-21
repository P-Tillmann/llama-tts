#!/usr/bin/env python3
# Convert a Piper voice (.onnx + .onnx.json) to a GGUF for llama-tts (--mmproj).
# Needs: gguf (from gguf-py in this repo), numpy, plus onnx (Piper) or torch (Kokoro).
#
# Layout notes (ggml ne order is the reverse of the numpy shape):
# - Conv1d k>1 [OC, IC, K] and ConvTranspose1d [IC, OC, K] are kept as in PyTorch.
# - 1x1 convs are squeezed to [OC, IC] so they are plain matmuls.
# - depthwise convs (duration predictor) are stored as [K, C], one row per tap.
# - attn_rel_v is stored transposed, [dk, 2w+1].
# - The 4 channel flips of the flow are folded into the weights: flows applied after an odd number of
#   flips get pre input channels and post output channels reversed. The runtime then alternates
#   a -= f(b) and b -= f(a) on the two halves without any flip.

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

import gguf

KEY = "clip.gen.audio."


def load_onnx(path):
    m = onnx.load(path)
    g = m.graph
    w = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    # weight-normed convs of the flow are folded into anonymous initializers; name them from the bias
    for n in g.node:
        if n.op_type == "Conv" and len(n.input) > 2 and n.input[2].endswith(".bias"):
            wname = n.input[2][: -len(".bias")] + ".weight"
            if wname not in w:
                w[wname] = w[n.input[1]]
    attrs = {}
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose") and n.input[1] in w:
            a = {at.name: helper_attr(at) for at in n.attribute}
            name = n.input[2][: -len(".bias")] + ".weight" if len(n.input) > 2 and n.input[2].endswith(".bias") else n.input[1]
            attrs[name] = a
    # token embedding: the Gather that reads graph input "input" (named "sid" in single-speaker exports)
    emb = [n.input[0] for n in g.node if n.op_type == "Gather" and len(n.input) == 2 and n.input[1] == "input"]
    if len(emb) != 1:
        sys.exit("token embedding not found")
    w["__tok_embd"] = w[emb[0]]
    # exp(-dp.flows.0.logs) is constant-folded into the Mul after Sub(x, dp.flows.0.m)
    sub = [n for n in g.node if n.op_type == "Sub" and "dp.flows.0.m" in n.input]
    if len(sub) != 1:
        sys.exit("dp.flows.0 ElementwiseAffine not found")
    mul = [n for n in g.node if n.op_type == "Mul" and sub[0].output[0] in n.input]
    other = [i for i in mul[0].input if i != sub[0].output[0]][0]
    if other in w:
        w["__dp_ea_scale"] = w[other]
    elif "dp.flows.0.logs" in w:
        w["__dp_ea_scale"] = np.exp(-w["dp.flows.0.logs"])
    else:
        sys.exit("dp.flows.0 scale not found")
    return w, attrs


def helper_attr(at):
    return list(at.ints) if at.ints else (at.i if at.type == onnx.AttributeProto.INT else at.f)


def count(w, fmt):
    n = 0
    while fmt.format(n) in w:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="voice .onnx, the .onnx.json must be next to it")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--outtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--model-card", help="MODEL_CARD file, stored in general.description")
    args = ap.parse_args()

    cfg = json.loads(Path(args.onnx + ".json").read_text(encoding="utf-8"))
    w, attrs = load_onnx(args.onnx)
    tensors = {}

    def put(name, arr, f32=False):
        assert name not in tensors, name
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if args.outtype == "f16" and not f32 and arr.ndim >= 2:
            arr = arr.astype(np.float16)
        tensors[name] = arr

    def put_conv(name, src, f32=False, has_bias=True):
        k = w[src + ".weight"]
        put(name + ".weight", k[:, :, 0] if k.shape[2] == 1 else k, f32)
        if has_bias:
            put(name + ".bias", w[src + ".bias"])

    def check_dilation(src, expect):
        d = attrs[src + ".weight"].get("dilations", [1])[0]
        if d != expect:
            sys.exit(f"{src}: dilation {d}, expected {expect}")

    # text encoder
    tok = w["__tok_embd"]
    n_vocab, n_embd = tok.shape
    put("a.gen.enc.token_embd.weight", tok)
    enc = "enc_p.encoder."
    n_layer = count(w, enc + "attn_layers.{}.conv_q.weight")
    rel = w[enc + "attn_layers.0.emb_rel_k"]
    if rel.shape[0] != 1:
        sys.exit("only heads_share=True is supported")
    d_head = rel.shape[2]
    n_head = n_embd // d_head
    n_ff, _, k_ffn = w[enc + "ffn_layers.0.conv_1.weight"].shape
    for il in range(n_layer):
        p = f"a.gen.enc.blk.{il}."
        for src, dst in (("conv_q", "attn_q"), ("conv_k", "attn_k"), ("conv_v", "attn_v"), ("conv_o", "attn_out")):
            put_conv(p + dst, f"{enc}attn_layers.{il}.{src}")
        put(p + "attn_rel_k", w[f"{enc}attn_layers.{il}.emb_rel_k"][0], f32=True)
        put(p + "attn_rel_v", w[f"{enc}attn_layers.{il}.emb_rel_v"][0].T, f32=True)
        put(p + "ln1.weight", w[f"{enc}norm_layers_1.{il}.gamma"])
        put(p + "ln1.bias", w[f"{enc}norm_layers_1.{il}.beta"])
        put_conv(p + "ffn_up", f"{enc}ffn_layers.{il}.conv_1")
        put_conv(p + "ffn_down", f"{enc}ffn_layers.{il}.conv_2")
        put(p + "ln2.weight", w[f"{enc}norm_layers_2.{il}.gamma"])
        put(p + "ln2.bias", w[f"{enc}norm_layers_2.{il}.beta"])
    put_conv("a.gen.enc.proj", "enc_p.proj")
    n_proj = w["enc_p.proj.weight"].shape[0] // 2

    n_spk = 1
    if "emb_g.weight" in w:
        n_spk = w["emb_g.weight"].shape[0]
        put("a.gen.spk_embd.weight", w["emb_g.weight"])

    # stochastic duration predictor
    if "dp.flows.1.pre.weight" in w or "dp.conv_1.weight" in w:
        sys.exit("unsupported duration predictor layout")

    def put_dds(dst, src):
        n = count(w, src + ".convs_sep.{}.weight")
        for il in range(n):
            check_dilation(f"{src}.convs_sep.{il}", w[src + ".convs_sep.0.weight"].shape[2] ** il)
            sep = w[f"{src}.convs_sep.{il}.weight"]
            put(f"{dst}.dds.{il}.conv.weight", sep[:, 0, :].T, f32=True)
            put(f"{dst}.dds.{il}.conv.bias", w[f"{src}.convs_sep.{il}.bias"])
            put_conv(f"{dst}.dds.{il}.conv1x1", f"{src}.convs_1x1.{il}")
            put(f"{dst}.dds.{il}.norm1.weight", w[f"{src}.norms_1.{il}.gamma"])
            put(f"{dst}.dds.{il}.norm1.bias", w[f"{src}.norms_1.{il}.beta"])
            put(f"{dst}.dds.{il}.norm2.weight", w[f"{src}.norms_2.{il}.gamma"])
            put(f"{dst}.dds.{il}.norm2.bias", w[f"{src}.norms_2.{il}.beta"])

    put_conv("a.gen.dp.pre", "dp.pre")
    if "dp.cond.weight" in w:
        put_conv("a.gen.dp.cond", "dp.cond")
    put_dds("a.gen.dp", "dp.convs")
    put_conv("a.gen.dp.proj", "dp.proj")
    # reverse order skips the unused dp.flows.1, see StochasticDurationPredictor.forward
    dp_flows = sorted((int(m.group(1)) for k in w for m in [re.match(r"dp\.flows\.(\d+)\.pre\.weight$", k)] if m), reverse=True)
    n_bins = None
    for i, f in enumerate(dp_flows):
        put_conv(f"a.gen.dp.flow.{i}.pre", f"dp.flows.{f}.pre")
        put_dds(f"a.gen.dp.flow.{i}", f"dp.flows.{f}.convs")
        put_conv(f"a.gen.dp.flow.{i}.proj", f"dp.flows.{f}.proj", f32=True)
        n_bins = (w[f"dp.flows.{f}.proj.weight"].shape[0] + 1) // 3
    if n_bins != 10:
        sys.exit(f"spline bins {n_bins}, expected 10")
    put("a.gen.dp.ea.m", w["dp.flows.0.m"].reshape(-1))
    put("a.gen.dp.ea.scale", w["__dp_ea_scale"].reshape(-1))

    # flow, in application order (reverse of the module list)
    flows = sorted((int(m.group(1)) for k in w for m in [re.match(r"flow\.flows\.(\d+)\.pre\.weight$", k)] if m), reverse=True)
    for j, f in enumerate(flows):
        src, dst = f"flow.flows.{f}", f"a.gen.flow.{j}"
        pre_w, post_w, post_b = w[src + ".pre.weight"][:, :, 0], w[src + ".post.weight"][:, :, 0], w[src + ".post.bias"]
        if j % 2 == 0:
            pre_w, post_w, post_b = pre_w[:, ::-1], post_w[::-1, :], post_b[::-1]
        put(dst + ".pre.weight", pre_w)
        put(dst + ".pre.bias", w[src + ".pre.bias"])
        put(dst + ".post.weight", post_w)
        put(dst + ".post.bias", post_b)
        if src + ".enc.cond_layer.weight" in w:
            put_conv(dst + ".wn.cond", src + ".enc.cond_layer")
        for il in range(count(w, src + ".enc.in_layers.{}.weight")):
            check_dilation(f"{src}.enc.in_layers.{il}", 1)
            put_conv(f"{dst}.wn.{il}.in", f"{src}.enc.in_layers.{il}")
            put_conv(f"{dst}.wn.{il}.res_skip", f"{src}.enc.res_skip_layers.{il}")

    # hifi-gan decoder
    put_conv("a.gen.dec.conv_pre", "dec.conv_pre")
    if "dec.cond.weight" in w:
        put_conv("a.gen.dec.cond", "dec.cond")
    n_up = count(w, "dec.ups.{}.weight")
    for i in range(n_up):
        k = w[f"dec.ups.{i}.weight"].shape[2]
        a = attrs[f"dec.ups.{i}.weight"]
        u, pad = a["strides"][0], a["pads"][0]
        if k != 2 * u or pad != (k - u) // 2:
            sys.exit(f"dec.ups.{i}: kernel {k} stride {u} pad {pad} not supported")
        put_conv(f"a.gen.dec.up.{i}", f"dec.ups.{i}")
    n_res = count(w, "dec.resblocks.{}.convs1.0.weight") or count(w, "dec.resblocks.{}.convs.0.weight")
    rb_type = 1 if "dec.resblocks.0.convs1.0.weight" in w else 2
    dil = {1: [(1, 3, 5)] * 3, 2: [(1, 2), (2, 6), (3, 12)]}[rb_type]
    n_kernels = n_res // n_up
    for r in range(n_res):
        src = f"dec.resblocks.{r}"
        c1 = "convs1" if rb_type == 1 else "convs"
        for d, dv in enumerate(dil[r % n_kernels]):
            check_dilation(f"{src}.{c1}.{d}", dv)
            put_conv(f"a.gen.dec.res.{r}.c1.{d}", f"{src}.{c1}.{d}")
            if rb_type == 1:
                put_conv(f"a.gen.dec.res.{r}.c2.{d}", f"{src}.convs2.{d}")
    put_conv("a.gen.dec.conv_post", "dec.conv_post", has_bias=False)

    # metadata
    inf = cfg.get("inference", {})
    pmap = cfg["phoneme_id_map"]
    symbols = [""] * n_vocab
    for ph, ids in pmap.items():
        if len(ids) != 1 or not 0 <= ids[0] < n_vocab or symbols[ids[0]]:
            sys.exit(f"phoneme {ph!r}: unsupported id mapping {ids}")
        symbols[ids[0]] = ph
    name = Path(args.onnx).name.removesuffix(".onnx")
    gw = gguf.GGUFWriter(args.outfile, "clip")
    gw.add_name(name)
    gw.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16" else gguf.LlamaFileType.ALL_F32)
    if args.model_card:
        gw.add_description(Path(args.model_card).read_text(encoding="utf-8"))
    gw.add_bool("clip.has_vision_encoder", False)
    gw.add_bool("clip.has_audio_encoder", False)
    gw.add_bool("clip.has_gen_audio_encoder", True)
    gw.add_string(KEY + "projector_type", "vits")
    gw.add_string(KEY + "model_variant", cfg.get("audio", {}).get("quality", ""))
    gw.add_uint32(KEY + "embedding_length", n_embd)
    gw.add_uint32(KEY + "feed_forward_length", n_ff)
    gw.add_uint32(KEY + "block_count", n_layer)
    gw.add_uint32(KEY + "projection_dim", n_proj)
    gw.add_uint32(KEY + "attention.head_count", n_head)
    gw.add_float32(KEY + "attention.layer_norm_epsilon", 1e-5)
    gw.add_uint32(KEY + "sample_rate", cfg["audio"]["sample_rate"])
    gw.add_float32(KEY + "noise_scale", float(inf.get("noise_scale", 0.667)))
    gw.add_float32(KEY + "length_scale", float(inf.get("length_scale", 1.0)))
    gw.add_float32(KEY + "noise_scale_w", float(inf.get("noise_w", 0.8)))
    gw.add_array(KEY + "symbols", symbols)
    if n_spk > 1:
        spk = [""] * n_spk
        for s, i in cfg.get("speaker_id_map", {}).items():
            spk[i] = s
        gw.add_array(KEY + "speakers", spk)
    gw.add_string(KEY + "phoneme_type", cfg.get("phoneme_type", "espeak"))
    gw.add_string(KEY + "espeak_voice", cfg.get("espeak", {}).get("voice", ""))

    for n, a in tensors.items():
        gw.add_tensor(n, a)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    n_par = sum(a.size for a in tensors.values())
    print(f"{args.outfile}: {len(tensors)} tensors, {n_par/1e6:.2f}M params, vocab {n_vocab}, embd {n_embd}, ff {n_ff}, "
          f"layers {n_layer}, heads {n_head}, proj {n_proj}, speakers {n_spk}, dp flows {len(dp_flows)}, flows {len(flows)}, "
          f"ups {n_up}, resblock type {rb_type} x{n_kernels}, sr {cfg['audio']['sample_rate']}")


if __name__ == "__main__":
    main()
