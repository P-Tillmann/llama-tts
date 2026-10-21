#pragma once

#include "../clip-graph.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct clip_graph_siglip : clip_graph {
    clip_graph_siglip(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4v : clip_graph {
    clip_graph_gemma4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
    bool support_batch() const override { return true; }
};

struct clip_graph_gemma4uv : clip_graph {
    clip_graph_gemma4uv(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_pixtral : clip_graph {
    clip_graph_pixtral(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseek4v : clip_graph {
    clip_graph_deepseek4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen2vl : clip_graph {
    clip_graph_qwen2vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_inp_with_temporal_merge();
};

struct clip_graph_qwen3vl : clip_graph_qwen2vl {
    clip_graph_qwen3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph_qwen2vl(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_ling3vl : clip_graph_qwen3vl {
    clip_graph_ling3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph_qwen3vl(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minimax_m3 : clip_graph {
    clip_graph_minimax_m3(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * apply_rope(ggml_tensor * x, ggml_tensor * pos_h, ggml_tensor * pos_w);
};

struct clip_graph_mimovl : clip_graph {
    clip_graph_mimovl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    // Force F32 mat-mul accumulation to avoid F16 overflow in the FFN down-proj
    // when the mmproj is stored in F16 (the source weights are BF16; downcasting
    // to F16 reduces dynamic range below the SwiGLU output magnitude on the last few layers).
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_step3vl : clip_graph {
    clip_graph_step3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_youtuvl : clip_graph {
    clip_graph_youtuvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_yasa2 : clip_graph {
    clip_graph_yasa2(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * layer_norm_channels(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b, float eps = 1e-6f);
    ggml_tensor * convnext_grn(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b);
};

struct clip_graph_minicpmv : clip_graph {
    clip_graph_minicpmv(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minicpmv4_6 : clip_graph {
    clip_graph_minicpmv4_6(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_internvl : clip_graph {
    clip_graph_internvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    bool support_batch() const override { return true; }
};

struct clip_graph_nemotron_v2_vl : clip_graph {
    clip_graph_nemotron_v2_vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_nemotron_omni : clip_graph {
    clip_graph_nemotron_omni(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_inp_with_temporal_merge();
};

struct clip_graph_llama4 : clip_graph {
    clip_graph_llama4(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimivl : clip_graph {
    clip_graph_kimivl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_paddleocr : clip_graph {
    clip_graph_paddleocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_dotsocr : clip_graph {
    clip_graph_dotsocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_dots3note_a : clip_graph {
    clip_graph_dots3note_a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_cogvlm : clip_graph {
    clip_graph_cogvlm(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llava : clip_graph {
    clip_graph_llava(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_whisper_enc : clip_graph {
    clip_graph_whisper_enc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

// causal whisper-like encoder with a sliding window KV cache (voxtral realtime)
struct clip_graph_voxtral_rt : clip_graph {
    clip_enc_state * state; // nullptr during warmup, the cache is then a graph input
    clip_graph_voxtral_rt(clip_ctx * ctx, const clip_image_f32 & img, clip_enc_state * state) : clip_graph(ctx, img), state(state) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseekocr : clip_graph {
    clip_graph_deepseekocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_sam(ggml_tensor * inp); // build the SAM model
    bool support_batch() const override { return true; }
};

struct clip_graph_deepseekocr2 : clip_graph_deepseekocr {
    clip_graph_deepseekocr2(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph_deepseekocr(ctx, img) {}
    ggml_cgraph * build() override; // reuses build_sam() from base
    bool support_batch() const override { return true; }
};

struct clip_graph_conformer : clip_graph {
    clip_graph_conformer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_granite_speech : clip_graph {
    clip_graph_granite_speech(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_muse_spark_audio : clip_graph {
    clip_graph_muse_spark_audio(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4a : clip_graph {
    clip_graph_gemma4a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_gemma4ua : clip_graph {
    clip_graph_gemma4ua(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_glm4v : clip_graph {
    clip_graph_glm4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_hunyuanvl : clip_graph {
    clip_graph_hunyuanvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mobilenetv5 : clip_graph {
    clip_graph_mobilenetv5(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * rms_norm_2d(
        ggml_tensor * inp,
        ggml_tensor * weight,
        float eps = 1e-6f);

    ggml_tensor* pad_same_2d(
        ggml_tensor* inp,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int dilation_h = 1,
        int dilation_w = 1);

    ggml_tensor * build_edge_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_inverted_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_mobilenet_attn(
        ggml_tensor * inp,
        const mobilenetv5_block & block);
};

struct clip_graph_qwen3a : clip_graph {
    clip_graph_qwen3a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mimo_audio : clip_graph {
    clip_graph_mimo_audio(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen3tts_spkenc : clip_graph {
    clip_graph_qwen3tts_spkenc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * conv1d_same(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation) const;
    ggml_tensor * res2net(ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const;
    ggml_tensor * se_block(ggml_tensor * x, const clip_layer & layer) const;
    ggml_tensor * se_res2net_block(ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const;
    ggml_tensor * attentive_stats_pool(ggml_tensor * x) const;
};

struct clip_graph_qwen3tts_gen : clip_graph {
    clip_graph_qwen3tts_gen(clip_ctx * ctx, const clip_image_f32 & img, clip_gen_process_type gen_process, int top_k, float top_p)
        : clip_graph(ctx, img), gen_process(gen_process), top_k(top_k), top_p(top_p) {}
    ggml_cgraph * build() override;

    // which sub-graph build() constructs, fixed at graph-build time
    clip_gen_process_type gen_process;

    // sampling params, fixed at graph-build time (GEN_CODE only)
    int   top_k;
    float top_p;

    //
    // code_gen: backbone hidden state + sampled code0 -> 16 RVQ codes
    // MTP-style code predictor, one token per codebook
    //
    struct code_gen : clip_graph {
        code_gen(const clip_graph & parent, int top_k, float top_p)
            : clip_graph(parent), top_k(top_k), top_p(top_p) {}
        ggml_cgraph * build() override { GGML_ABORT("call prefill()/step() instead"); }

        int   top_k;
        float top_p;

        ggml_tensor * cache_set(ggml_tensor * cache, int row_idx, ggml_tensor * value) const;
        ggml_tensor * do_sampling(ggml_tensor * logits, ggml_tensor * inp_rand) const;

        ggml_tensor * const_i32(ggml_tensor * anchor, float value) const;
        ggml_tensor * causal_mask_row(int64_t n_kv_pad, int pos) const;
        ggml_tensor * project_in(ggml_tensor * cur) const;

        ggml_tensor * layer_forward(
                ggml_tensor * cur,
                const clip_layer & layer,
                ggml_tensor * inp_pos,
                ggml_tensor * kq_mask,
                ggml_tensor *& k_cache_layer,
                ggml_tensor *& v_cache_layer,
                int64_t n_kv_pad,
                int pos,
                int il) const;

        void prefill(
                std::vector<ggml_tensor *> & k_cache,
                std::vector<ggml_tensor *> & v_cache,
                ggml_tensor *& out_code_cache,
                ggml_tensor * h_state,
                ggml_tensor * code0_embd,
                ggml_tensor * inp_rand) const;

        ggml_tensor * step(
                std::vector<ggml_tensor *> & k_cache,
                std::vector<ggml_tensor *> & v_cache,
                ggml_tensor * out_code_cache,
                ggml_tensor * inp_rand,
                int step_idx) const;
    };

    //
    // code2wav: RVQ codes -> raw PCM (quantizer + pre_conv + pre_transformer + upsample + DAC).
    //
    struct code2wav : clip_graph {
        code2wav(const clip_graph & parent) : clip_graph(parent) {}
        ggml_cgraph * build() override { GGML_ABORT("call decode() instead"); }

        // state_in: previous call's persisted state, by slot name (see list_c2w_state_slots())
        std::map<std::string, ggml_tensor *> state_in;
        // state_out: this call's state to persist, added to the graph outputs by build()
        mutable std::vector<std::pair<std::string, ggml_tensor *>> state_out;

        // stateful conv ops: read/update their state via state_in/state_out[state_name]
        ggml_tensor * causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, const std::string & state_name) const;
        ggml_tensor * causal_conv1d_dw(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const std::string & state_name) const;
        ggml_tensor * causal_conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & state_name) const;
        ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const;

        ggml_tensor * quant_decode(ggml_tensor * inp_codes) const;
        ggml_tensor * tfm_layer_forward(ggml_tensor * cur, const clip_layer & layer, int il) const;
        ggml_tensor * convnext_block(ggml_tensor * x, const clip_code2wav::upsample_block & blk, const std::string & state_prefix) const;
        ggml_tensor * dac_res_unit(ggml_tensor * x, const clip_code2wav::dac_res & res, int dilation, const std::string & state_name) const;

        // inp_codes [1, n_codes] I32 -> this frame's audio samples [n_samples] F32, clamped to [-1, 1]
        ggml_tensor * decode(ggml_tensor * inp_codes) const;
    };
};

//
// pocket-tts: SEANet convolution stack, shared by the voice encoder and the mimi decoder.
// stateless unless state_in is populated: convs then pad instead of carrying left-context.
//
struct clip_graph_pockettts_seanet : clip_graph {
    clip_graph_pockettts_seanet(const clip_graph & parent) : clip_graph(parent) {}
    ggml_cgraph * build() override { GGML_ABORT("call encode()/decode() instead"); }

    // per-call streaming state, keyed by slot name (see list_pockettts_state_slots)
    std::map<std::string, ggml_tensor *> state_in;
    mutable std::vector<std::pair<std::string, ggml_tensor *>> state_out;

    ggml_tensor * conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation,
                         bool pad_replicate = false, const std::string & state_name = "") const;
    ggml_tensor * conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride,
                                   const std::string & state_name = "") const;
    ggml_tensor * res_unit(ggml_tensor * x, const clip_seanet::stage & stage, int dilation,
                           const std::string & state_prefix = "") const;

    // x: [T, C] -> [T / hop, dim]
    ggml_tensor * encode(ggml_tensor * x) const;
    // x: [T, dim] -> [T * hop, 1], streams when state_in is populated
    ggml_tensor * decode(ggml_tensor * x) const;
};

// mimi encoder + speaker_proj: reference waveform -> voice conditioning rows
struct clip_graph_pockettts_spkenc : clip_graph {
    clip_graph_pockettts_spkenc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * tfm_layer_forward(ggml_tensor * cur, const clip_layer & layer, ggml_tensor * inp_pos, ggml_tensor * kq_mask, int il) const;
};

//
// pocket-tts generation:
// GEN_CODE = flow-matching decoder + end-of-speech head, one latent per call
// GEN_WAV  = mimi decoder, a window of latents -> PCM
//
struct clip_graph_pockettts_gen : clip_graph {
    clip_graph_pockettts_gen(clip_ctx * ctx, const clip_image_f32 & img, clip_gen_process_type gen_process, int n_step, int n_frames)
        : clip_graph(ctx, img), gen_process(gen_process), n_step(n_step), n_frames(n_frames) {}
    ggml_cgraph * build() override;

    clip_gen_process_type gen_process;
    int n_step;   // lsd_decode steps, fixed at graph-build time
    int n_frames; // GEN_WAV only: number of latents to decode

    // AdaLN modulation: x * (1 + scale) + shift
    ggml_tensor * modulate(ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) const;
    ggml_tensor * time_embed(const clip_flow_net::time_embd & te, float t) const;
    ggml_tensor * flow_forward(ggml_tensor * cond, ggml_tensor * x, float s, float t) const;
};

//
// vits (piper): phoneme ids -> PCM, two graphs per call, see clip_encode_vits()
// stage 0: text encoder + stochastic duration predictor -> "out_stats", "out_logw"
// stage 1: frame-expanded stats -> flow -> hifi-gan -> "out_audio"
//
struct clip_graph_vits : clip_graph {
    clip_graph_vits(clip_ctx * ctx, const clip_image_f32 & img, int stage, int n_tokens, int n_frames, float noise_scale, float noise_scale_w)
        : clip_graph(ctx, img), stage(stage), n_tokens(n_tokens), n_frames(n_frames), noise_scale(noise_scale), noise_scale_w(noise_scale_w) {}
    ggml_cgraph * build() override;

    int   stage;
    int   n_tokens;
    int   n_frames;      // stage 1 only
    float noise_scale;   // prior noise, stage 1
    float noise_scale_w; // duration noise, stage 0

    ggml_tensor * conv1x1(ggml_tensor * x, const clip_vits::wb & c) const;
    ggml_tensor * conv1d(ggml_tensor * x, const clip_vits::wb & c, int dilation) const;
    ggml_tensor * conv1d_dw(ggml_tensor * x, const clip_vits::wb & c, int dilation) const;
    ggml_tensor * conv_transpose1d(ggml_tensor * x, const clip_vits::wb & c) const;
    ggml_tensor * layer_norm(ggml_tensor * x, const clip_vits::wb & n) const;
    ggml_tensor * rows(ggml_tensor * x, int64_t i0, int64_t n) const;
    ggml_tensor * rel_attn(ggml_tensor * x, int il) const;
    ggml_tensor * dds(const clip_vits::dds_net & net, ggml_tensor * x) const;
    ggml_tensor * spline_inverse(ggml_tensor * x, ggml_tensor * h, int64_t n_filt) const;
    ggml_tensor * wavenet(const clip_vits::coupling & c, ggml_tensor * x, ggml_tensor * g) const;
    ggml_tensor * resblock(ggml_tensor * x, const clip_vits::resblock & rb, int j) const;
    void build_text(ggml_tensor * g);
    void build_wave(ggml_tensor * g);
};

//
// kokoro (StyleTTS2 prosody + iSTFTNet): phoneme ids -> PCM, two graphs per call, see clip_encode_kokoro()
// stage 0: ALBERT, duration encoder, durations, text encoder -> "out_dur", "out_d", "out_t_en" (channel-major)
// stage 1: frame-expanded features -> F0/N, decoder, harmonic source, generator, iSTFT -> "out_audio" (time-major)
//
struct clip_graph_kokoro : clip_graph {
    clip_graph_kokoro(clip_ctx * ctx, const clip_image_f32 & img, int stage, int n_tokens, int n_frames, float speed, float noise_scale)
        : clip_graph(ctx, img), stage(stage), n_tokens(n_tokens), n_frames(n_frames), speed(speed), noise_scale(noise_scale) {}
    ggml_cgraph * build() override;

    int   stage;
    int   n_tokens;
    int   n_frames;    // stage 1 only, frames of 600 samples
    float speed;       // stage 0, divides the durations
    float noise_scale; // stage 1, scales the harmonic source noise

    ggml_tensor * inp_style = nullptr;

    ggml_tensor * linear(ggml_tensor * x, const clip_kokoro::wb & l) const;
    ggml_tensor * conv1d_cm(ggml_tensor * x, const clip_kokoro::wb & c) const;
    ggml_tensor * bilstm(ggml_tensor * x, const clip_kokoro::lstm & l) const;
    ggml_tensor * ada_layer_norm(ggml_tensor * x, const clip_kokoro::ada & a, ggml_tensor * s) const;
    ggml_tensor * style(int64_t offset) const;

    ggml_tensor * conv1d_tm(ggml_tensor * x, const clip_kokoro::wb & c, int stride, int pad, int dilation) const;
    ggml_tensor * conv_transpose1d_tm(ggml_tensor * x, const clip_kokoro::wb & c, int stride, int pad) const;
    ggml_tensor * adain(ggml_tensor * x, const clip_kokoro::ada & a, ggml_tensor * s) const;
    ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * ialpha) const;
    ggml_tensor * adain_resblk(ggml_tensor * x, const clip_kokoro::adain_resblk & b, ggml_tensor * s) const;
    ggml_tensor * adain_resblock1(ggml_tensor * x, const clip_kokoro::adain_resblock1 & b, ggml_tensor * s) const;

    void build_text();
    void build_wave();
};

// one persisted state buffer used by code2wav, see qwen3tts-gen.cpp
struct c2w_state_slot {
    std::string name;
    int64_t     ne0;
    int64_t     ne1;
};
std::vector<c2w_state_slot> list_c2w_state_slots(const clip_hparams & hparams, const clip_model & model);

// same, for the streaming mimi decoder (pocket-tts GEN_WAV)
std::vector<c2w_state_slot> list_pockettts_state_slots(const clip_hparams & hparams, const clip_model & model);

struct clip_graph_kimik25 : clip_graph {
    clip_graph_kimik25(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * resize_position_embeddings_3d(uint32_t interpolation_mode);
};

struct clip_graph_parakeet : clip_graph {
    clip_graph_parakeet(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_exaone4_5 : clip_graph {
    clip_graph_exaone4_5(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_granite4_vision : clip_graph {
    clip_graph_granite4_vision(clip_ctx * ctx, const clip_image_f32 & img)
        : clip_graph(ctx, img),
          anyres(img.anyres),
          n_tiles(img.ny() / img.nx()),
          tile_side(img.nx() / patch_size) {}

    ggml_cgraph * build() override;

private:
    // the input image is a stack of tiles on the Y axis: [overview, tile(0,0), tile(0,1), ...]
    const clip_image_f32::anyres_info anyres;
    const int n_tiles;
    const int tile_side; // patches per tile side

    ggml_tensor * build_tile_inp();
    ggml_tensor * gather(ggml_tensor * src, const std::string & name, int idx_len);
    ggml_tensor * interp_down(ggml_tensor * src, int side, int new_side);
    ggml_tensor * build_block(const qf_block & blk, ggml_tensor * h, int bid,
                              int spatial_offset, int image_side, int window_side,
                              int query_side, float qformer_eps);

    ggml_tensor * build_newline_row(ggml_context * ctx0);
    ggml_tensor * build_anyres_assembly(ggml_tensor * cur, int out_side);
};

struct clip_graph_muse_glimmer : clip_graph {
    clip_graph_muse_glimmer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};
