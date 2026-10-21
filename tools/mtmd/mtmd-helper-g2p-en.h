#pragma once

// English G2P for Kokoro, ported from misaki en.py (Apache-2.0, https://github.com/hexgrad/misaki)
// internal to mtmd-helper

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class g2p_en {
public:
    // phonemes in the misaki format for one word, false if there are none
    using fallback_t = std::function<bool(const std::string & word, std::string & phonemes)>;

    // lexicon from mtmd_gen_audio_info: the first n_gold entries are preferred, "word\tPOS" are heteronym variants
    g2p_en(const char * const * words, const char * const * phonemes, int32_t n, int32_t n_gold, bool british);
    ~g2p_en();

    // one paragraph to phoneme chunks of at most 510 symbols, like KPipeline.en_tokenize() in kokoro
    std::vector<std::string> phonemize(const std::string & text, const fallback_t & fallback) const;

    // espeak IPA (with ^ ties) to Kokoro phonemes, misaki espeak.py
    // lang is the Kokoro language code: 'a' American English, 'b' British English, others use the generic mapping
    static std::string from_espeak(const std::string & ipa, char lang);

private:
    struct impl;
    std::unique_ptr<impl> p;
};
