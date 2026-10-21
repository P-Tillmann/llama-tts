#include "models.h"

#include <algorithm>
#include <cmath>
#include <vector>

// the text side is channel-major (ne0 = channels), the frame side is time-major (ne0 = time):
// instance norms over time are then plain ggml_norm, and convs need no transposes

// bidirectional LSTM recurrence on the CPU, one direction per thread
// src[0] xg [8H, T] = W_ih x + b for both directions, src[1] w_hh [H, 4H, 2], dst [2H, T]
static void kokoro_lstm_cpu(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * xg  = dst->src[0];
    const ggml_tensor * whh = dst->src[1];
    const int64_t H = whh->ne[0];
    const int64_t T = xg->ne[1];
    std::vector<float> h(H), c(H), g(4 * H);
    for (int dir = ith; dir < 2; dir += nth) {
        const float * w = (const float *) ((const char *) whh->data + dir * whh->nb[2]);
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        for (int64_t s = 0; s < T; s++) {
            const int64_t t = dir == 0 ? s : T - 1 - s;
            const float * xt = (const float *) ((const char *) xg->data + t * xg->nb[1]) + dir * 4 * H;
            for (int64_t r = 0; r < 4 * H; r++) {
                const float * wr = w + r * H;
                // independent partial sums so the compiler can vectorize
                float acc[8] = {};
                int64_t k = 0;
                for (; k + 8 <= H; k += 8) {
                    for (int u = 0; u < 8; u++) {
                        acc[u] += wr[k + u] * h[k + u];
                    }
                }
                float sum = xt[r];
                for (; k < H; k++) {
                    sum += wr[k] * h[k];
                }
                for (int u = 0; u < 8; u++) {
                    sum += acc[u];
                }
                g[r] = sum;
            }
            float * out = (float *) ((char *) dst->data + t * dst->nb[1]) + dir * H;
            for (int64_t k = 0; k < H; k++) {
                const float ig = 1.0f / (1.0f + expf(-g[k]));
                const float fg = 1.0f / (1.0f + expf(-g[H + k]));
                const float gg = tanhf(g[2 * H + k]);
                const float og = 1.0f / (1.0f + expf(-g[3 * H + k]));
                c[k]   = fg * c[k] + ig * gg;
                h[k]   = og * tanhf(c[k]);
                out[k] = h[k];
            }
        }
    }
}

// cumulative sum along ne0 with a double accumulator, like torch.cumsum on the CPU; the phase grows to ~1e6 rad
static void kokoro_cumsum_f64_cpu(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * /*userdata*/) {
    const int64_t n = a->ne[0];
    for (int64_t r = ith; r < ggml_nrows(a); r += nth) {
        const float * x = (const float *) ((const char *) a->data + r * a->nb[1]);
        float * y = (float *) ((char *) dst->data + r * dst->nb[1]);
        double acc = 0.0;
        for (int64_t i = 0; i < n; i++) {
            acc += x[i];
            y[i] = (float) acc;
        }
    }
}

// x300 linear upsampling along ne0 (align_corners=False) with the arithmetic of torch upsample_linear1d on the CPU,
// including its fused multiply-adds: the phase is large, so one rounding step changes the sine
static void kokoro_upsample300_cpu(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * a = dst->src[0];
    const int64_t n_in  = a->ne[0];
    const int64_t n_out = dst->ne[0];
    const float scale = (float) (1.0 / 300.0);
    for (int64_t r = ith; r < ggml_nrows(a); r += nth) {
        const float * x = (const float *) ((const char *) a->data + r * a->nb[1]);
        float * y = (float *) ((char *) dst->data + r * dst->nb[1]);
        for (int64_t i = 0; i < n_out; i++) {
            float src = std::fmaf(scale, (float) i + 0.5f, -0.5f);
            src = src < 0.0f ? 0.0f : src;
            const int64_t i0 = std::min<int64_t>((int64_t) src, n_in - 1);
            const int64_t i1 = i0 + (i0 < n_in - 1 ? 1 : 0);
            const float l1 = std::min(std::max(src - (float) i0, 0.0f), 1.0f);
            const float l0 = 1.0f - l1;
            y[i] = std::fmaf(l0, x[i0], l1 * x[i1]);
        }
    }
}

ggml_tensor * clip_graph_kokoro::linear(ggml_tensor * x, const clip_kokoro::wb & l) const {
    return ggml_add(ctx0, ggml_mul_mat(ctx0, l.w, x), l.b);
}

// "same" conv, x [C, T] channel-major, w [K, IC, OC]
ggml_tensor * clip_graph_kokoro::conv1d_cm(ggml_tensor * x, const clip_kokoro::wb & c) const {
    const int64_t K  = c.w->ne[0];
    const int64_t IC = c.w->ne[1];
    const int64_t OC = c.w->ne[2];
    ggml_tensor * xt  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * col = ggml_im2col(ctx0, c.w, xt, 1, 0, (int) (K - 1) / 2, 0, 1, 0, false, GGML_TYPE_F32);
    col = ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]);
    return ggml_add(ctx0, ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, c.w, K * IC, OC), col), c.b);
}

// x [IN, T] -> [2H, T], forward then backward hidden states
ggml_tensor * clip_graph_kokoro::bilstm(ggml_tensor * x, const clip_kokoro::lstm & l) const {
    ggml_tensor * xg = ggml_add(ctx0, ggml_mul_mat(ctx0, l.w_ih, x), l.b);
    ggml_tensor * args[] = { xg, l.w_hh };
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, 2 * l.w_hh->ne[0], x->ne[1], 1, 1, args, 2, kokoro_lstm_cpu, 2, nullptr);
}

// layer norm over channels without affine, then gamma(s) * x + beta(s)
ggml_tensor * clip_graph_kokoro::ada_layer_norm(ggml_tensor * x, const clip_kokoro::ada & a, ggml_tensor * s) const {
    x = ggml_norm(ctx0, x, 1e-5f);
    return ggml_add(ctx0, ggml_mul(ctx0, x, linear(s, a.gamma)), linear(s, a.beta));
}

// style vector of the voice: [0, 128) for the decoder, [128, 256) for the prosody predictor
ggml_tensor * clip_graph_kokoro::style(int64_t offset) const {
    ggml_tensor * voices = model.kokoro.voices;
    voices = ggml_reshape_2d(ctx0, voices, voices->ne[0], voices->ne[1] * voices->ne[2]);
    ggml_tensor * st = ggml_get_rows(ctx0, voices, inp_style);
    return ggml_cont(ctx0, ggml_view_2d(ctx0, st, 128, 1, st->nb[1], offset * sizeof(float)));
}

// x [T, IC] time-major, w [K, IC, OC] -> [T', OC]
// conv_2d_direct with H = 1 runs im2col in tiles, so long inputs do not need a full im2col buffer
ggml_tensor * clip_graph_kokoro::conv1d_tm(ggml_tensor * x, const clip_kokoro::wb & c, int stride, int pad, int dilation) const {
    const int64_t K  = c.w->ne[0];
    const int64_t IC = c.w->ne[1];
    const int64_t OC = c.w->ne[2];
    ggml_tensor * cur = ggml_conv_2d_direct(ctx0, ggml_reshape_4d(ctx0, c.w, K, 1, IC, OC), ggml_reshape_4d(ctx0, x, x->ne[0], 1, IC, 1),
                                            stride, 1, pad, 0, dilation, 1);
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], OC);
    if (c.b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, c.b, 1, OC));
    }
    return cur;
}

// x [T, IC] time-major, w [K, OC, IC] -> [(T - 1) * stride + K - 2 * pad, OC]
ggml_tensor * clip_graph_kokoro::conv_transpose1d_tm(ggml_tensor * x, const clip_kokoro::wb & c, int stride, int pad) const {
    const int64_t K  = c.w->ne[0];
    const int64_t OC = c.w->ne[1];
    const int64_t IC = c.w->ne[2];
    ggml_tensor * w2  = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, c.w, K * OC, IC)));
    ggml_tensor * xc  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * cur = ggml_col2im_1d(ctx0, ggml_mul_mat(ctx0, w2, xc), stride, (int) OC, pad);
    return ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, c.b, 1, OC));
}

// instance norm over time, then gamma(s) * x + beta(s)
ggml_tensor * clip_graph_kokoro::adain(ggml_tensor * x, const clip_kokoro::ada & a, ggml_tensor * s) const {
    const int64_t C = x->ne[1];
    ggml_tensor * gamma = ggml_reshape_2d(ctx0, linear(s, a.gamma), 1, C);
    ggml_tensor * beta  = ggml_reshape_2d(ctx0, linear(s, a.beta),  1, C);
    x = ggml_norm(ctx0, x, 1e-5f);
    return ggml_add(ctx0, ggml_mul(ctx0, x, gamma), beta);
}

// x + sin(alpha * x)^2 / alpha
ggml_tensor * clip_graph_kokoro::snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * ialpha) const {
    return ggml_add(ctx0, x, ggml_mul(ctx0, ggml_sqr(ctx0, ggml_sin(ctx0, ggml_mul(ctx0, x, alpha))), ialpha));
}

// AdainResBlk1d, optional x2 upsampling (depthwise transposed conv on the residual, nearest on the shortcut)
ggml_tensor * clip_graph_kokoro::adain_resblk(ggml_tensor * x, const clip_kokoro::adain_resblk & b, ggml_tensor * s) const {
    const bool upsample = b.pool.w != nullptr;

    ggml_tensor * h = ggml_leaky_relu(ctx0, adain(x, b.norm1, s), 0.2f, false);
    if (upsample) {
        // k3 s2 p1 output_padding 1: y[2m] = w1 x[m], y[2m+1] = w2 x[m] + w0 x[m+1], x[T] = 0
        const int64_t T = h->ne[0];
        const int64_t C = h->ne[1];
        auto tap = [&](int k) {
            return ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, b.pool.w, C, k * C * sizeof(float)), 1, C);
        };
        ggml_tensor * next = ggml_cont(ctx0, ggml_view_2d(ctx0, h, T - 1, C, h->nb[1], sizeof(float)));
        next = ggml_pad_ext(ctx0, next, 0, 1, 0, 0, 0, 0, 0, 0);
        ggml_tensor * even = ggml_mul(ctx0, h, tap(1));
        ggml_tensor * odd  = ggml_add(ctx0, ggml_mul(ctx0, h, tap(2)), ggml_mul(ctx0, next, tap(0)));
        h = ggml_concat(ctx0, ggml_reshape_3d(ctx0, even, 1, T, C), ggml_reshape_3d(ctx0, odd, 1, T, C), 0);
        h = ggml_add(ctx0, ggml_reshape_2d(ctx0, h, 2 * T, C), ggml_reshape_2d(ctx0, b.pool.b, 1, C));
    }
    h = conv1d_tm(h, b.conv1, 1, 1, 1);
    h = ggml_leaky_relu(ctx0, adain(h, b.norm2, s), 0.2f, false);
    h = conv1d_tm(h, b.conv2, 1, 1, 1);

    ggml_tensor * sc = x;
    if (upsample) {
        sc = ggml_interpolate(ctx0, sc, 2 * sc->ne[0], sc->ne[1], 1, 1, GGML_SCALE_MODE_NEAREST);
    }
    if (b.conv1x1_w) {
        sc = conv1d_tm(sc, { b.conv1x1_w, nullptr }, 1, 0, 1);
    }
    return ggml_scale(ctx0, ggml_add(ctx0, h, sc), 1.0f / sqrtf(2.0f));
}

// AdaINResBlock1 of the generator: AdaIN, Snake, dilated conv, AdaIN, Snake, conv, residual
ggml_tensor * clip_graph_kokoro::adain_resblock1(ggml_tensor * x, const clip_kokoro::adain_resblock1 & b, ggml_tensor * s) const {
    static const int dil[3] = { 1, 3, 5 };
    for (int d = 0; d < 3; d++) {
        const int K = (int) b.c1[d].w->ne[0];
        ggml_tensor * t = snake(adain(x, b.n1[d], s), b.alpha1[d], b.ialpha1[d]);
        t = conv1d_tm(t, b.c1[d], 1, dil[d] * (K - 1) / 2, dil[d]);
        t = snake(adain(t, b.n2[d], s), b.alpha2[d], b.ialpha2[d]);
        t = conv1d_tm(t, b.c2[d], 1, (K - 1) / 2, 1);
        x = ggml_add(ctx0, t, x);
    }
    return x;
}

void clip_graph_kokoro::build_text() {
    const auto & ko = model.kokoro;

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_ids, "inp_codes");
    ggml_set_input(inp_ids);

    ggml_tensor * s_pro = style(128);
    ggml_tensor * s_rep = ggml_repeat_4d(ctx0, s_pro, 128, n_tokens, 1, 1);

    // plbert: ALBERT with one shared layer
    ggml_tensor * x = ggml_get_rows(ctx0, ko.bert_tok_embd, inp_ids);
    x = ggml_add(ctx0, x, ggml_view_2d(ctx0, ko.bert_pos_embd, ko.bert_pos_embd->ne[0], n_tokens, ko.bert_pos_embd->nb[1], 0));
    x = build_norm(x, ko.bert_embd_norm.w, ko.bert_embd_norm.b, NORM_TYPE_NORMAL, eps, -1);
    x = linear(x, ko.bert_embd_proj);
    const auto & layer = ko.bert_layer;
    for (int il = 0; il < n_layer; il++) {
        ggml_tensor * q = ggml_reshape_3d(ctx0, linear(x, { layer.q_w, layer.q_b }), d_head, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, linear(x, { layer.k_w, layer.k_b }), d_head, n_head, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, linear(x, { layer.v_w, layer.v_b }), d_head, n_head, n_tokens);
        ggml_tensor * attn = build_attn(layer.o_w, layer.o_b, q, k, v, nullptr, kq_scale, il);
        x = build_norm(ggml_add(ctx0, x, attn), layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        // gelu_new (tanh approximation) written out, the CPU ggml_gelu uses an f16 table
        ggml_tensor * ff = linear(x, { layer.ff_up_w, layer.ff_up_b });
        ggml_tensor * inner = ggml_add(ctx0, ff, ggml_scale(ctx0, ggml_mul(ctx0, ggml_sqr(ctx0, ff), ff), 0.044715f));
        inner = ggml_tanh(ctx0, ggml_scale(ctx0, inner, sqrtf(2.0f / (float) M_PI)));
        ff = ggml_mul(ctx0, ggml_scale(ctx0, ff, 0.5f), ggml_scale_bias(ctx0, inner, 1.0f, 1.0f));
        ff = linear(ff, { layer.ff_down_w, layer.ff_down_b });
        x = build_norm(ggml_add(ctx0, x, ff), layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
    }
    cb(x, "bert", -1);
    x = linear(x, ko.bert_proj);
    cb(x, "bert_proj", -1);

    // duration encoder: LSTM, AdaLayerNorm, style re-concatenated after each layer
    x = ggml_concat(ctx0, x, s_rep, 0);
    for (int i = 0; i < 3; i++) {
        x = ada_layer_norm(bilstm(x, ko.dur_enc_lstm[i]), ko.dur_enc_norm[i], s_pro);
        cb(x, "dur_enc_ln", i);
        x = ggml_concat(ctx0, x, s_rep, 0);
    }
    ggml_set_name(x, "out_d");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // durations: sum of sigmoids over max_dur, divided by speed (division, as the reference rounds after it)
    ggml_tensor * dur = bilstm(x, ko.dur_lstm);
    cb(dur, "dur_lstm", -1);
    dur = ggml_sum_rows(ctx0, ggml_sigmoid(ctx0, linear(dur, ko.dur_proj)));
    if (speed != 1.0f) {
        dur = ggml_div(ctx0, dur, ggml_fill(ctx0, ggml_view_1d(ctx0, dur, 1, 0), speed));
    }
    ggml_set_name(dur, "out_dur");
    ggml_set_output(dur);
    ggml_build_forward_expand(gf, dur);

    // text encoder
    ggml_tensor * t = ggml_get_rows(ctx0, ko.text_tok_embd, inp_ids);
    for (int i = 0; i < 3; i++) {
        t = conv1d_cm(t, ko.text_conv[i]);
        t = build_norm(t, ko.text_norm[i].w, ko.text_norm[i].b, NORM_TYPE_NORMAL, 1e-5f, -1);
        t = ggml_leaky_relu(ctx0, t, 0.2f, false);
    }
    t = bilstm(t, ko.text_lstm);
    ggml_set_name(t, "out_t_en");
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
}

void clip_graph_kokoro::build_wave() {
    const auto & ko = model.kokoro;
    const int64_t n_d    = ko.dur_lstm.w_ih->ne[0];
    const int64_t n_t_en = 2 * ko.text_lstm.w_hh->ne[0];
    const int64_t n_samples = 600 * (int64_t) n_frames;

    ggml_tensor * inp_d = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_d, n_tokens);
    ggml_set_name(inp_d, "inp_d");
    ggml_set_input(inp_d);
    ggml_tensor * inp_t_en = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_t_en, n_tokens);
    ggml_set_name(inp_t_en, "inp_t_en");
    ggml_set_input(inp_t_en);
    // token index of each frame, from the rounded durations
    ggml_tensor * inp_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(inp_idx, "inp_idx");
    ggml_set_input(inp_idx);
    // standard normal noise of the 9 harmonics, [9, L] like randn_like() in the reference
    ggml_tensor * inp_noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 9, n_samples);
    ggml_set_name(inp_noise, "inp_noise");
    ggml_set_input(inp_noise);

    // constants from the host: harmonic numbers 1..9, sample rate, STFT and iSTFT bases, 1 / window envelope
    ggml_tensor * inp_harm = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 9);
    ggml_set_name(inp_harm, "inp_harm");
    ggml_set_input(inp_harm);
    ggml_tensor * inp_sr = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(inp_sr, "inp_sr");
    ggml_set_input(inp_sr);
    ggml_tensor * inp_stft = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 20, 1, 22);
    ggml_set_name(inp_stft, "inp_stft");
    ggml_set_input(inp_stft);
    ggml_tensor * inp_istft = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 22, 20);
    ggml_set_name(inp_istft, "inp_istft");
    ggml_set_input(inp_istft);
    ggml_tensor * inp_env = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_samples);
    ggml_set_name(inp_env, "inp_env");
    ggml_set_input(inp_env);

    ggml_tensor * s_pro = style(128);
    ggml_tensor * s_dec = style(0);

    // prosody: F0 and energy curves at twice the frame rate
    ggml_tensor * en  = ggml_get_rows(ctx0, inp_d, inp_idx);
    ggml_tensor * asr = ggml_get_rows(ctx0, inp_t_en, inp_idx);
    cb(en, "en", -1);
    ggml_tensor * sh = bilstm(en, ko.shared_lstm);
    cb(sh, "shared", -1);
    sh = ggml_cont(ctx0, ggml_transpose(ctx0, sh));
    ggml_tensor * f0 = sh;
    ggml_tensor * nn = sh;
    for (int i = 0; i < 3; i++) {
        f0 = adain_resblk(f0, ko.f0[i], s_pro);
        nn = adain_resblk(nn, ko.n[i],  s_pro);
        cb(f0, "f0_blk", i);
        cb(nn, "n_blk", i);
    }
    f0 = conv1d_tm(f0, ko.f0_proj, 1, 0, 1);
    nn = conv1d_tm(nn, ko.n_proj,  1, 0, 1);
    cb(f0, "f0_curve", -1);
    cb(nn, "n_curve", -1);

    // decoder
    ggml_tensor * asr_tm = ggml_cont(ctx0, ggml_transpose(ctx0, asr));
    cb(asr_tm, "asr", -1);
    ggml_tensor * f0d = conv1d_tm(f0, ko.dec_f0_conv, 2, 1, 1);
    ggml_tensor * nd  = conv1d_tm(nn, ko.dec_n_conv,  2, 1, 1);
    cb(f0d, "dec_f0", -1);
    cb(nd, "dec_n", -1);
    ggml_tensor * x = ggml_concat(ctx0, ggml_concat(ctx0, asr_tm, f0d, 1), nd, 1);
    x = adain_resblk(x, ko.dec_encode, s_dec);
    cb(x, "dec_encode", -1);
    ggml_tensor * asr_res = conv1d_tm(asr_tm, ko.dec_asr_res, 1, 0, 1);
    cb(asr_res, "asr_res", -1);
    for (int i = 0; i < 4; i++) {
        x = ggml_concat(ctx0, ggml_concat(ctx0, ggml_concat(ctx0, x, asr_res, 1), f0d, 1), nd, 1);
        x = adain_resblk(x, ko.dec_decode[i], s_dec);
        cb(x, "dec_decode", i);
    }

    // harmonic source, computed at twice the frame rate first: in SineGen._f02sine() the 1/300 linear
    // downsample of the nearest-upsampled f0 is exactly the frame value, and the random initial phase has no effect
    ggml_tensor * rad = ggml_div(ctx0, ggml_mul(ctx0, ggml_repeat_4d(ctx0, f0, f0->ne[0], 9, 1, 1), inp_harm), inp_sr);
    rad = ggml_sub(ctx0, rad, ggml_floor(ctx0, rad));
    ggml_tensor * phase = ggml_map_custom1(ctx0, rad, kokoro_cumsum_f64_cpu, GGML_N_TASKS_MAX, nullptr);
    phase = ggml_scale(ctx0, ggml_scale(ctx0, ggml_scale(ctx0, phase, 2.0f), (float) M_PI), 300.0f);
    ggml_tensor * args[] = { phase };
    phase = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_samples, 9, 1, 1, args, 1, kokoro_upsample300_cpu, GGML_N_TASKS_MAX, nullptr);
    phase = ggml_cont(ctx0, ggml_transpose(ctx0, phase));
    ggml_tensor * sines = ggml_scale(ctx0, ggml_sin(ctx0, phase), 0.1f);
    // voiced where f0 > 10; noise std 0.003 when voiced, 0.1 / 3 when not
    ggml_tensor * uv = ggml_step(ctx0, ggml_scale_bias(ctx0, f0, 1.0f, -10.0f));
    uv = ggml_reshape_2d(ctx0, ggml_interpolate(ctx0, uv, n_samples, 1, 1, 1, GGML_SCALE_MODE_NEAREST), 1, n_samples);
    ggml_tensor * namp = ggml_add(ctx0, ggml_scale(ctx0, uv, 0.003f), ggml_scale(ctx0, ggml_scale_bias(ctx0, uv, -1.0f, 1.0f), (float) (0.1 / 3)));
    ggml_tensor * noise = ggml_scale(ctx0, inp_noise, noise_scale);
    cb(noise, "sine_noise", -1);
    ggml_tensor * sw = ggml_add(ctx0, ggml_mul(ctx0, sines, uv), ggml_mul(ctx0, noise, namp));
    cb(sw, "sine_waves", -1);
    ggml_tensor * har = ggml_mul_mat(ctx0, ko.l_linear.w, sw);
    har = ggml_reshape_2d(ctx0, ggml_tanh(ctx0, ggml_add(ctx0, har, ko.l_linear.b)), n_samples, 1);
    cb(har, "har_source", -1);

    // STFT (n_fft 20, hop 5, centered): windowed DFT basis as a strided conv, then magnitude and phase
    ggml_tensor * spec = conv1d_tm(ggml_pad_reflect_1d(ctx0, har, 10, 10), { inp_stft, nullptr }, 5, 0, 1);
    const int64_t n_stft = spec->ne[0];
    ggml_tensor * re = ggml_view_2d(ctx0, spec, n_stft, 11, spec->nb[1], 0);
    ggml_tensor * im = ggml_view_2d(ctx0, spec, n_stft, 11, spec->nb[1], 11 * spec->nb[1]);
    ggml_tensor * mag = ggml_sqrt(ctx0, ggml_add(ctx0, ggml_sqr(ctx0, re), ggml_sqr(ctx0, im)));
    ggml_tensor * pha = ggml_atan2(ctx0, im, re);
    cb(mag, "har_spec", -1);
    cb(pha, "har_phase", -1);
    ggml_tensor * hs = ggml_concat(ctx0, mag, pha, 1);

    // generator
    const int up_stride[2] = { 10, 6 };
    for (int i = 0; i < 2; i++) {
        x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        ggml_tensor * xs = i == 0 ? conv1d_tm(hs, ko.noise_conv[0], 6, 3, 1) : conv1d_tm(hs, ko.noise_conv[1], 1, 0, 1);
        cb(xs, "gen_noise_conv", i);
        xs = adain_resblock1(xs, ko.noise_res[i], s_dec);
        cb(xs, "gen_noise_res", i);
        const int K = (int) ko.ups[i].w->ne[0];
        x = conv_transpose1d_tm(x, ko.ups[i], up_stride[i], (K - up_stride[i]) / 2);
        cb(x, "gen_up", i);
        if (i == 1) {
            x = ggml_pad_reflect_1d(ctx0, x, 1, 0);
        }
        x = ggml_add(ctx0, x, xs);
        ggml_tensor * sum = nullptr;
        for (int j = 0; j < 3; j++) {
            ggml_tensor * r = adain_resblock1(x, ko.res[3 * i + j], s_dec);
            sum = sum ? ggml_add(ctx0, sum, r) : r;
        }
        x = ggml_scale(ctx0, sum, 1.0f / 3.0f);
        cb(x, "gen_stage", i);
    }
    x = ggml_leaky_relu(ctx0, x, 0.01f, false);
    x = conv1d_tm(x, ko.conv_post, 1, 3, 1);
    cb(x, "gen_post", -1);

    // iSTFT: magnitude exp(.), phase sin(.), one-sided inverse DFT basis with the window, overlap-add
    ggml_tensor * mag2 = ggml_exp(ctx0, ggml_view_2d(ctx0, x, n_stft, 11, x->nb[1], 0));
    ggml_tensor * ph2  = ggml_sin(ctx0, ggml_view_2d(ctx0, x, n_stft, 11, x->nb[1], 11 * x->nb[1]));
    ggml_tensor * ri = ggml_concat(ctx0, ggml_mul(ctx0, mag2, ggml_cos(ctx0, ph2)), ggml_mul(ctx0, mag2, ggml_sin(ctx0, ph2)), 1);
    ggml_tensor * frames = ggml_mul_mat(ctx0, inp_istft, ggml_cont(ctx0, ggml_transpose(ctx0, ri)));
    ggml_tensor * y = ggml_col2im_1d(ctx0, frames, 5, 1, 10);
    y = ggml_mul(ctx0, ggml_reshape_1d(ctx0, y, n_samples), inp_env);
    ggml_set_name(y, "out_audio");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
}

ggml_cgraph * clip_graph_kokoro::build() {
    inp_style = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_style, "inp_style");
    ggml_set_input(inp_style);
    if (stage == 0) {
        build_text();
    } else {
        build_wave();
    }
    return gf;
}
