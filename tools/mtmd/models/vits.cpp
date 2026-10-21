#include "models.h"

// activations are channel-major (ne0 = channels, ne1 = time), except in the decoder:
// its convs run on long sequences and use the time-major direct conv (ne0 = time)

ggml_tensor * clip_graph_vits::conv1x1(ggml_tensor * x, const clip_vits::wb & c) const {
    ggml_tensor * cur = ggml_mul_mat(ctx0, c.w, x);
    if (c.b) {
        cur = ggml_add(ctx0, cur, c.b);
    }
    return cur;
}

// "same" padding, w is [K, IC, OC]; the f32 im2col keeps full precision (ggml_conv_1d uses f16)
ggml_tensor * clip_graph_vits::conv1d(ggml_tensor * x, const clip_vits::wb & c, int dilation) const {
    const int64_t K  = c.w->ne[0];
    const int64_t IC = c.w->ne[1];
    const int64_t OC = c.w->ne[2];
    GGML_ASSERT(K % 2 == 1);
    ggml_tensor * xt  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * col = ggml_im2col(ctx0, c.w, xt, 1, 0, dilation * (K - 1) / 2, 0, dilation, 0, false, GGML_TYPE_F32);
    col = ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]);
    ggml_tensor * cur = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, c.w, K * IC, OC), col);
    if (c.b) {
        cur = ggml_add(ctx0, cur, c.b);
    }
    return cur;
}

// time-major "same" conv, x [T, IC], w [K, IC, OC] -> [T, OC], no transposes
// faster than conv_2d_direct for the small channel counts of the piper decoder
ggml_tensor * clip_graph_vits::conv1d_tm(ggml_tensor * x, const clip_vits::wb & c, int dilation) const {
    const int64_t K  = c.w->ne[0];
    const int64_t IC = c.w->ne[1];
    const int64_t OC = c.w->ne[2];
    GGML_ASSERT(K % 2 == 1);
    ggml_tensor * col = ggml_im2col(ctx0, c.w, x, 1, 0, dilation * (int) (K - 1) / 2, 0, dilation, 0, false, GGML_TYPE_F32);
    col = ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]);
    // the weight is src1 here, which the CPU matmul needs in f32
    ggml_tensor * w = c.w->type == GGML_TYPE_F32 ? c.w : ggml_cast(ctx0, c.w, GGML_TYPE_F32);
    ggml_tensor * cur = ggml_mul_mat(ctx0, col, ggml_reshape_2d(ctx0, w, K * IC, OC));
    if (c.b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, c.b, 1, OC));
    }
    return cur;
}

// depthwise "same" conv, w is [C, K]: one row of per-channel taps per k
ggml_tensor * clip_graph_vits::conv1d_dw(ggml_tensor * x, const clip_vits::wb & c, int dilation) const {
    const int64_t C   = x->ne[0];
    const int64_t T   = x->ne[1];
    const int64_t K   = c.w->ne[1];
    const int     pad = dilation * (K - 1) / 2;
    ggml_tensor * xp  = ggml_pad_ext(ctx0, x, 0, 0, pad, pad, 0, 0, 0, 0);
    ggml_tensor * cur = nullptr;
    for (int64_t k = 0; k < K; k++) {
        ggml_tensor * xs = ggml_view_2d(ctx0, xp, C, T, xp->nb[1], k * dilation * xp->nb[1]);
        ggml_tensor * t  = ggml_mul(ctx0, xs, ggml_view_1d(ctx0, c.w, C, k * c.w->nb[1]));
        cur = cur ? ggml_add(ctx0, cur, t) : t;
    }
    return ggml_add(ctx0, cur, c.b);
}

// time-major in and out, w is [K, OC, IC], stride K/2 and padding (K - stride)/2 so T_out = T * stride (checked at conversion)
ggml_tensor * clip_graph_vits::conv_transpose1d(ggml_tensor * x, const clip_vits::wb & c) const {
    const int64_t K  = c.w->ne[0];
    const int64_t OC = c.w->ne[1];
    const int64_t IC = c.w->ne[2];
    const int stride = K / 2;
    ggml_tensor * w2  = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, c.w, K * OC, IC)));
    ggml_tensor * xc  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * cur = ggml_col2im_1d(ctx0, ggml_mul_mat(ctx0, w2, xc), stride, OC, (K - stride) / 2);
    return ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, c.b, 1, OC));
}

ggml_tensor * clip_graph_vits::layer_norm(ggml_tensor * x, const clip_vits::wb & n) const {
    return build_norm(x, n.w, n.b, NORM_TYPE_NORMAL, eps, -1);
}

// channels [i0, i0 + n)
ggml_tensor * clip_graph_vits::rows(ggml_tensor * x, int64_t i0, int64_t n) const {
    return ggml_cont(ctx0, ggml_view_2d(ctx0, x, n, x->ne[1], x->nb[1], i0 * x->nb[0]));
}

// self-attention with relative position embeddings shared by the heads, window w
ggml_tensor * clip_graph_vits::rel_attn(ggml_tensor * x, int il) const {
    const auto & layer = model.vits.enc_layers[il];
    ggml_tensor * rel_k = model.vits.enc_rel_k[il];
    ggml_tensor * rel_v = model.vits.enc_rel_v[il];
    const int64_t N     = x->ne[1];
    const int64_t n_rel = rel_k->ne[1];
    const int64_t win   = (n_rel - 1) / 2;

    ggml_tensor * q = ggml_scale(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, layer.q_w, x), layer.q_b), kq_scale);
    ggml_tensor * k = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.k_w, x), layer.k_b);
    ggml_tensor * v = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.v_w, x), layer.v_b);
    q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, q, d_head, n_head, N), 0, 2, 1, 3);
    k = ggml_permute(ctx0, ggml_reshape_3d(ctx0, k, d_head, n_head, N), 0, 2, 1, 3);
    v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, d_head, n_head, N), 1, 2, 0, 3));

    // relative logits [2w+1, N, n_head], padded or cut to the 2N-1 offsets of the reference
    ggml_tensor * rel = ggml_mul_mat(ctx0, rel_k, q);
    if (N - 1 >= win) {
        rel = ggml_pad_ext(ctx0, rel, N - 1 - win, N - 1 - win, 0, 0, 0, 0, 0, 0);
    } else {
        rel = ggml_cont(ctx0, ggml_view_3d(ctx0, rel, 2 * N - 1, N, n_head, rel->nb[1], rel->nb[2], (win - (N - 1)) * rel->nb[0]));
    }
    // rel -> abs: pos[i][j] = rel[i][j - i + N - 1], a view with row stride 2N-2
    ggml_tensor * pos = ggml_view_3d(ctx0, rel, N, N, n_head, (2 * N - 2) * rel->nb[0], rel->nb[2], (N - 1) * rel->nb[0]);

    ggml_tensor * kq = ggml_add(ctx0, ggml_mul_mat(ctx0, k, q), ggml_cont(ctx0, pos));
    kq = ggml_soft_max(ctx0, kq);
    ggml_tensor * out = ggml_mul_mat(ctx0, v, kq);

    // abs -> rel inside the window: relw[i][r] = kq[i][i + r - w], a view with row stride N+2w+1
    ggml_tensor * kqp  = ggml_pad_ext(ctx0, kq, win, win, 0, 0, 0, 0, 0, 0);
    ggml_tensor * relw = ggml_cont(ctx0, ggml_view_3d(ctx0, kqp, n_rel, N, n_head, (N + 2 * win + 1) * kqp->nb[0], kqp->nb[2], 0));
    out = ggml_add(ctx0, out, ggml_mul_mat(ctx0, rel_v, relw));

    out = ggml_cont_2d(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3), n_embd, N);
    return ggml_add(ctx0, ggml_mul_mat(ctx0, layer.o_w, out), layer.o_b);
}

// dilated depth-separable conv stack, dilation K^i
ggml_tensor * clip_graph_vits::dds(const clip_vits::dds_net & net, ggml_tensor * x) const {
    int dilation = 1;
    for (const auto & l : net.layers) {
        ggml_tensor * y = conv1d_dw(x, l.conv, dilation);
        y = ggml_gelu_erf(ctx0, layer_norm(y, l.norm1));
        y = conv1x1(y, l.conv1x1);
        y = ggml_gelu_erf(ctx0, layer_norm(y, l.norm2));
        x = ggml_add(ctx0, x, y);
        dilation *= (int) l.conv.w->ne[1];
    }
    return x;
}

// inverse rational-quadratic spline with linear tails, h is [3*n_bins - 1, N]
// bins are selected with a one-hot mask and gathered with sum_rows
ggml_tensor * clip_graph_vits::spline_inverse(ggml_tensor * x, ggml_tensor * h, int64_t n_filt) const {
    const int64_t n_bins = (h->ne[0] + 1) / 3;
    const float bound = 5.0f;
    const float min_bin = 1e-3f;
    const float scale = 1.0f / sqrtf((float) n_filt);

    ggml_tensor * one_row = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0);
    auto knots = [&](ggml_tensor * u) {
        ggml_tensor * a = ggml_soft_max(ctx0, ggml_scale(ctx0, u, scale));
        a = ggml_scale_bias(ctx0, a, 1.0f - min_bin * n_bins, min_bin);
        ggml_tensor * c = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, a), 2.0f * bound, -bound);
        // first knot is -bound, the last one is forced to +bound
        c = ggml_concat(ctx0, ggml_fill(ctx0, one_row, -bound), rows(c, 0, n_bins - 1), 0);
        return ggml_concat(ctx0, c, ggml_fill(ctx0, one_row, bound), 0);
    };
    ggml_tensor * cw = knots(rows(h, 0, n_bins));
    ggml_tensor * ch = knots(rows(h, n_bins, n_bins));
    ggml_tensor * widths  = ggml_sub(ctx0, rows(cw, 1, n_bins), rows(cw, 0, n_bins));
    ggml_tensor * heights = ggml_sub(ctx0, rows(ch, 1, n_bins), rows(ch, 0, n_bins));

    // derivatives at the knots, the outer ones are 1 (linear tails)
    const float d_out = (float) std::log(std::exp(1.0 - min_bin) - 1.0);
    ggml_tensor * dc = ggml_fill(ctx0, one_row, d_out);
    ggml_tensor * d  = ggml_concat(ctx0, ggml_concat(ctx0, dc, rows(h, 2 * n_bins, n_bins - 1), 0), dc, 0);
    d = ggml_scale_bias(ctx0, ggml_softplus(ctx0, d), 1.0f, min_bin);

    // clamp first, the spline branch must stay finite outside of the bound
    ggml_tensor * xc = ggml_clamp(ctx0, x, -bound, bound);

    // bin = number of knots <= x, minus 1; the last knot is moved up by 1e-6 like the reference
    ggml_tensor * ch_s = ggml_concat(ctx0, rows(ch, 0, n_bins), ggml_fill(ctx0, one_row, bound + 1e-6f), 0);
    ggml_tensor * ge   = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_sub(ctx0, ch_s, xc)), -1.0f, 1.0f);
    ggml_tensor * onehot = ggml_sub(ctx0, rows(ge, 0, n_bins), rows(ge, 1, n_bins));
    auto gather = [&](ggml_tensor * t) {
        return ggml_sum_rows(ctx0, ggml_mul(ctx0, onehot, t));
    };
    ggml_tensor * in_cw    = gather(rows(cw, 0, n_bins));
    ggml_tensor * in_w     = gather(widths);
    ggml_tensor * in_ch    = gather(rows(ch, 0, n_bins));
    ggml_tensor * in_h     = gather(heights);
    ggml_tensor * in_delta = gather(ggml_div(ctx0, heights, widths));
    ggml_tensor * in_d     = gather(rows(d, 0, n_bins));
    ggml_tensor * in_d1    = gather(rows(d, 1, n_bins));

    // solve the quadratic for theta, see rational_quadratic_spline() in the reference
    ggml_tensor * t  = ggml_sub(ctx0, xc, in_ch);
    ggml_tensor * s2 = ggml_sub(ctx0, ggml_add(ctx0, in_d, in_d1), ggml_scale(ctx0, in_delta, 2.0f));
    ggml_tensor * ts = ggml_mul(ctx0, t, s2);
    ggml_tensor * qa = ggml_add(ctx0, ts, ggml_mul(ctx0, in_h, ggml_sub(ctx0, in_delta, in_d)));
    ggml_tensor * qb = ggml_sub(ctx0, ggml_mul(ctx0, in_h, in_d), ts);
    ggml_tensor * qc = ggml_neg(ctx0, ggml_mul(ctx0, in_delta, t));
    ggml_tensor * disc = ggml_sub(ctx0, ggml_sqr(ctx0, qb), ggml_scale(ctx0, ggml_mul(ctx0, qa, qc), 4.0f));
    disc = ggml_relu(ctx0, disc);
    ggml_tensor * root = ggml_div(ctx0, ggml_scale(ctx0, qc, 2.0f), ggml_sub(ctx0, ggml_neg(ctx0, qb), ggml_sqrt(ctx0, disc)));
    ggml_tensor * out  = ggml_add(ctx0, ggml_mul(ctx0, root, in_w), in_cw);

    // identity outside of [-bound, bound]
    ggml_tensor * inside = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_abs(ctx0, x), 1.0f, -bound)), -1.0f, 1.0f);
    return ggml_add(ctx0, ggml_mul(ctx0, inside, out), ggml_mul(ctx0, ggml_scale_bias(ctx0, inside, -1.0f, 1.0f), x));
}

// non-causal WaveNet body of a coupling layer, gated tanh * sigmoid with optional speaker conditioning
ggml_tensor * clip_graph_vits::wavenet(const clip_vits::coupling & c, ggml_tensor * x, ggml_tensor * g) const {
    const int64_t hid = x->ne[0];
    ggml_tensor * gc  = g && c.wn_cond.w ? conv1x1(g, c.wn_cond) : nullptr;
    ggml_tensor * out = nullptr;
    const size_t n_layer_wn = c.wn_in.size();
    for (size_t il = 0; il < n_layer_wn; il++) {
        ggml_tensor * xi = conv1d(x, c.wn_in[il], 1);
        if (gc) {
            xi = ggml_add(ctx0, xi, ggml_view_1d(ctx0, gc, 2 * hid, il * 2 * hid * gc->nb[0]));
        }
        ggml_tensor * acts = ggml_mul(ctx0, ggml_tanh(ctx0, rows(xi, 0, hid)), ggml_sigmoid(ctx0, rows(xi, hid, hid)));
        ggml_tensor * rs   = conv1x1(acts, c.wn_res[il]);
        ggml_tensor * skip = rs;
        if (il + 1 < n_layer_wn) {
            x    = ggml_add(ctx0, x, rows(rs, 0, hid));
            skip = rows(rs, hid, hid);
        }
        out = out ? ggml_add(ctx0, out, skip) : skip;
    }
    return out;
}

// hifi-gan residual block (time-major); dilations are fixed by the piper configs (checked at conversion)
ggml_tensor * clip_graph_vits::resblock(ggml_tensor * x, const clip_vits::resblock & rb, int j) const {
    static const int dil_1[3]    = { 1, 3, 5 };
    static const int dil_2[3][2] = { { 1, 2 }, { 2, 6 }, { 3, 12 } };
    const bool type_1 = !rb.c2.empty();
    for (size_t d = 0; d < rb.c1.size(); d++) {
        ggml_tensor * t = ggml_leaky_relu(ctx0, x, 0.1f, false);
        t = conv1d_tm(t, rb.c1[d], type_1 ? dil_1[d] : dil_2[j][d]);
        if (type_1) {
            t = ggml_leaky_relu(ctx0, t, 0.1f, false);
            t = conv1d_tm(t, rb.c2[d], 1);
        }
        x = ggml_add(ctx0, x, t);
    }
    return x;
}

void clip_graph_vits::build_text(ggml_tensor * g) {
    const auto & vits = model.vits;

    ggml_tensor * inp_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_ids, "inp_codes");
    ggml_set_input(inp_ids);
    // standard normal, one row per channel
    ggml_tensor * inp_noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, 2);
    ggml_set_name(inp_noise, "inp_noise");
    ggml_set_input(inp_noise);

    ggml_tensor * x = ggml_get_rows(ctx0, vits.tok_embd, inp_ids);
    x = ggml_scale(ctx0, x, sqrtf((float) n_embd));
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = vits.enc_layers[il];
        x = layer_norm(ggml_add(ctx0, x, rel_attn(x, il)), { layer.ln_1_w, layer.ln_1_b });
        ggml_tensor * y = ggml_relu(ctx0, conv1d(x, { layer.ff_up_w, layer.ff_up_b }, 1));
        y = conv1d(y, { layer.ff_down_w, layer.ff_down_b }, 1);
        x = layer_norm(ggml_add(ctx0, x, y), { layer.ln_2_w, layer.ln_2_b });
    }
    cb(x, "enc_out", -1);

    ggml_tensor * stats = conv1x1(x, vits.enc_proj);
    ggml_set_name(stats, "out_stats");
    ggml_set_output(stats);
    ggml_build_forward_expand(gf, stats);

    // stochastic duration predictor, reverse direction
    ggml_tensor * h = conv1x1(x, vits.dp.pre);
    if (g && vits.dp.cond.w) {
        h = ggml_add(ctx0, h, conv1x1(g, vits.dp.cond));
    }
    h = conv1x1(dds(vits.dp, h), vits.dp.proj);
    cb(h, "dp_x", -1);
    const int64_t n_filt = h->ne[0];

    ggml_tensor * z0 = ggml_view_2d(ctx0, inp_noise, 1, n_tokens, sizeof(float), 0);
    ggml_tensor * z1 = ggml_view_2d(ctx0, inp_noise, 1, n_tokens, sizeof(float), n_tokens * sizeof(float));
    z0 = ggml_scale(ctx0, z0, noise_scale_w);
    z1 = ggml_scale(ctx0, z1, noise_scale_w);
    cb(z0, "dp_noise", 0);
    cb(z1, "dp_noise", 1);
    // each conv flow follows a channel flip
    for (const auto & f : vits.dp_flows) {
        std::swap(z0, z1);
        ggml_tensor * hf = ggml_add(ctx0, conv1x1(z0, f.pre), h);
        hf = conv1x1(dds(f, hf), f.proj);
        z1 = spline_inverse(z1, hf, n_filt);
    }
    std::swap(z0, z1);
    // elementwise affine, channel 0 only
    ggml_tensor * logw = ggml_sub(ctx0, z0, ggml_view_1d(ctx0, vits.dp_ea_m, 1, 0));
    logw = ggml_mul(ctx0, logw, ggml_view_1d(ctx0, vits.dp_ea_scale, 1, 0));
    ggml_set_name(logw, "out_logw");
    ggml_set_output(logw);
    ggml_build_forward_expand(gf, logw);
}

void clip_graph_vits::build_wave(ggml_tensor * g) {
    const auto & vits = model.vits;
    const int64_t n_stats = vits.enc_proj.w->ne[1];
    const int64_t n_ch    = n_stats / 2;

    ggml_tensor * inp_stats = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_stats, n_tokens);
    ggml_set_name(inp_stats, "inp_stats");
    ggml_set_input(inp_stats);
    // token index of each frame, from the rounded durations
    ggml_tensor * inp_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(inp_idx, "inp_idx");
    ggml_set_input(inp_idx);
    ggml_tensor * inp_noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ch, n_frames);
    ggml_set_name(inp_noise, "inp_noise");
    ggml_set_input(inp_noise);

    ggml_tensor * e = ggml_get_rows(ctx0, inp_stats, inp_idx);
    ggml_tensor * m    = rows(e, 0, n_ch);
    ggml_tensor * logs = rows(e, n_ch, n_ch);
    ggml_tensor * eps = ggml_scale(ctx0, inp_noise, noise_scale);
    cb(eps, "prior_noise", -1);
    ggml_tensor * z = ggml_add(ctx0, m, ggml_mul(ctx0, eps, ggml_exp(ctx0, logs)));
    cb(z, "z_p", -1);

    // flow, reverse direction; the flips are folded into the weights, so the halves alternate
    ggml_tensor * a = rows(z, 0, n_ch / 2);
    ggml_tensor * b = rows(z, n_ch / 2, n_ch / 2);
    for (size_t j = 0; j < vits.flows.size(); j++) {
        const auto & f = vits.flows[j];
        ggml_tensor * hf = conv1x1(j % 2 == 0 ? b : a, f.pre);
        hf = conv1x1(wavenet(f, hf, g), f.post);
        if (j % 2 == 0) {
            a = ggml_sub(ctx0, a, hf);
            cb(a, "flow_a", (int) j);
        } else {
            b = ggml_sub(ctx0, b, hf);
            cb(b, "flow_b", (int) j);
        }
    }
    z = ggml_concat(ctx0, a, b, 0);
    cb(z, "z", -1);

    // hifi-gan, time-major
    ggml_tensor * x = conv1d_tm(ggml_cont(ctx0, ggml_transpose(ctx0, z)), vits.dec_conv_pre, 1);
    cb(x, "dec_pre", -1);
    if (g && vits.dec_cond.w) {
        ggml_tensor * gc = conv1x1(g, vits.dec_cond);
        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, gc, 1, gc->ne[0]));
    }
    const int n_kernels = (int) (vits.dec_res.size() / vits.dec_ups.size());
    for (size_t i = 0; i < vits.dec_ups.size(); i++) {
        x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        x = conv_transpose1d(x, vits.dec_ups[i]);
        cb(x, "dec_up", (int) i);
        ggml_tensor * xs = nullptr;
        for (int j = 0; j < n_kernels; j++) {
            ggml_tensor * r = resblock(x, vits.dec_res[i * n_kernels + j], j);
            xs = xs ? ggml_add(ctx0, xs, r) : r;
        }
        x = ggml_scale(ctx0, xs, 1.0f / n_kernels);
        cb(x, "dec_stage", (int) i);
    }
    x = ggml_leaky_relu(ctx0, x, 0.01f, false);
    x = conv1d_tm(x, { vits.dec_conv_post_w, nullptr }, 1);
    cb(x, "dec_post", -1);
    x = ggml_tanh(ctx0, x);
    ggml_set_name(x, "out_audio");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
}

ggml_cgraph * clip_graph_vits::build() {
    ggml_tensor * g = nullptr;
    if (model.vits.spk_embd) {
        ggml_tensor * inp_sid = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(inp_sid, "inp_sid");
        ggml_set_input(inp_sid);
        g = ggml_get_rows(ctx0, model.vits.spk_embd, inp_sid);
    }
    if (stage == 0) {
        build_text(g);
    } else {
        build_wave(g);
    }
    return gf;
}
