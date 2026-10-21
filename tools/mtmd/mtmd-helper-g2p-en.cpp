// English G2P for Kokoro, ported from misaki en.py and kokoro pipeline.py (Apache-2.0):
// https://github.com/hexgrad/misaki, https://github.com/hexgrad/kokoro
// spaCy is replaced by a rule-based tokenizer and a small context tagger, heteronyms mostly get their default pronunciation
// num2words is replaced by our own number words

#include "mtmd-helper-g2p-en.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <unordered_map>

using ustr = std::u32string;
using stress_t = std::optional<double>;
using ps_t = std::optional<ustr>;

//
// strings
//

static ustr to_u32(const std::string & s) {
    ustr out;
    for (size_t i = 0; i < s.size(); ) {
        const unsigned char c = (unsigned char) s[i];
        char32_t cp;
        size_t n;
        if (c < 0x80)              { cp = c;        n = 1; }
        else if ((c >> 5) == 0x6)  { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 0xE)  { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; n = 4; }
        else                       { i++; continue; }
        if (i + n > s.size()) {
            break;
        }
        for (size_t k = 1; k < n; k++) {
            cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        }
        out += cp;
        i += n;
    }
    return out;
}

static std::string to_u8(const ustr & s) {
    std::string out;
    for (char32_t cp : s) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// letters: ASCII and Latin-1/Latin Extended-A/B, enough for English text
static bool is_letter(char32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0x24F && c != 0xD7 && c != 0xF7);
}
static bool is_upper(char32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7);
}
static bool is_lower_letter(char32_t c) {
    return is_letter(c) && !is_upper(c);
}
static char32_t lower_c(char32_t c) {
    return is_upper(c) ? c + 32 : c;
}
static char32_t upper_c(char32_t c) {
    return ((c >= 'a' && c <= 'z') || (c >= 0xE0 && c <= 0xFE && c != 0xF7)) ? c - 32 : c;
}
static bool is_digit_c(char32_t c) {
    return c >= '0' && c <= '9';
}
static bool is_space_c(char32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
static ustr lower(ustr s) {
    for (auto & c : s) {
        c = lower_c(c);
    }
    return s;
}
static ustr upper(ustr s) {
    for (auto & c : s) {
        c = upper_c(c);
    }
    return s;
}
// Python str.capitalize(): first upper, rest lower
static ustr capitalize(ustr s) {
    s = lower(s);
    if (!s.empty()) {
        s[0] = upper_c(s[0]);
    }
    return s;
}
static bool is_alpha(const ustr & s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), is_letter);
}
static bool is_digits(const ustr & s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), is_digit_c);
}
static bool has(const ustr & s, char32_t c) {
    return s.find(c) != ustr::npos;
}
static bool starts_with(const ustr & s, const ustr & p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
static bool ends_with(const ustr & s, const ustr & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
static ustr replace_all(ustr s, const ustr & from, const ustr & to) {
    for (size_t pos = s.find(from); pos != ustr::npos; pos = s.find(from, pos + to.size())) {
        s.replace(pos, from.size(), to);
    }
    return s;
}
static ustr remove_c(ustr s, char32_t c) {
    s.erase(std::remove(s.begin(), s.end(), c), s.end());
    return s;
}
static ustr head(const ustr & s, size_t n_drop) {
    return s.substr(0, s.size() - std::min(n_drop, s.size()));
}

//
// phonemes
//

static const char32_t PRIMARY   = U'ˈ';
static const char32_t SECONDARY = U'ˌ';
static const ustr STRESSES         = U"ˌˈ";
static const ustr DIPHTHONGS       = U"AIOQWYʤʧ";
static const ustr VOWELS           = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
static const ustr CONSONANTS       = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
static const ustr US_TAUS          = U"AIOWYiuæɑəɛɪɹʊʌ";
static const ustr PUNCTS           = U";:,.!?—…\"“”";
static const ustr NON_QUOTE_PUNCTS = U";:,.!?—…";
static const ustr SUBTOKEN_JUNKS   = U"',-._‘’/";

static int stress_weight(const ustr & ps) {
    int w = 0;
    for (char32_t c : ps) {
        w += has(DIPHTHONGS, c) ? 2 : 1;
    }
    return w;
}

static bool has_vowel(const ustr & ps) {
    return std::any_of(ps.begin(), ps.end(), [](char32_t c) { return has(VOWELS, c); });
}

// move each stress mark right before the next vowel
static ustr restress(const ustr & ps) {
    std::vector<std::pair<double, char32_t>> ips;
    for (size_t i = 0; i < ps.size(); i++) {
        ips.push_back({ (double) i, ps[i] });
    }
    for (size_t i = 0; i < ps.size(); i++) {
        if (has(STRESSES, ps[i])) {
            for (size_t j = i; j < ps.size(); j++) {
                if (has(VOWELS, ps[j])) {
                    ips[i].first = (double) j - 0.5;
                    break;
                }
            }
        }
    }
    std::sort(ips.begin(), ips.end());
    ustr out;
    for (const auto & p : ips) {
        out += p.second;
    }
    return out;
}

static ustr apply_stress(const ustr & ps, stress_t stress) {
    if (!stress) {
        return ps;
    }
    const double s = *stress;
    const bool no_stress = !has(ps, PRIMARY) && !has(ps, SECONDARY);
    if (s < -1) {
        return remove_c(remove_c(ps, PRIMARY), SECONDARY);
    }
    if (s == -1 || ((s == 0 || s == -0.5) && has(ps, PRIMARY))) {
        return replace_all(remove_c(ps, SECONDARY), ustr(1, PRIMARY), ustr(1, SECONDARY));
    }
    if ((s == 0 || s == 0.5 || s == 1) && no_stress) {
        return has_vowel(ps) ? restress(SECONDARY + ps) : ps;
    }
    if (s >= 1 && !has(ps, PRIMARY) && has(ps, SECONDARY)) {
        return replace_all(ps, ustr(1, SECONDARY), ustr(1, PRIMARY));
    }
    if (s > 1 && no_stress) {
        return has_vowel(ps) ? restress(PRIMARY + ps) : ps;
    }
    return ps;
}

//
// number words, compatible with num2words (lang EN) as used by misaki
//

static const char * const NUM_ONES[] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
};
static const char * const NUM_TENS[] = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
static const char * const NUM_SCALES[] = { "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion" };

static std::string num_cardinal(int64_t n) {
    if (n < 0) {
        return "minus " + num_cardinal(-n);
    }
    if (n < 20) {
        return NUM_ONES[n];
    }
    if (n < 100) {
        return std::string(NUM_TENS[n / 10]) + (n % 10 ? std::string("-") + NUM_ONES[n % 10] : "");
    }
    if (n < 1000) {
        return std::string(NUM_ONES[n / 100]) + " hundred" + (n % 100 ? " and " + num_cardinal(n % 100) : "");
    }
    // groups of three digits, highest first
    std::vector<int64_t> groups;
    for (int64_t m = n; m > 0; m /= 1000) {
        groups.push_back(m % 1000);
    }
    std::string out;
    for (int g = (int) groups.size() - 1; g >= 0; g--) {
        if (groups[g] == 0) {
            continue;
        }
        if (!out.empty()) {
            out += g == 0 && groups[0] < 100 ? " and " : ", ";
        }
        out += num_cardinal(groups[g]);
        if (g > 0) {
            out += std::string(" ") + NUM_SCALES[std::min<int>(g, 6)];
        }
    }
    return out;
}

static std::string num_ordinal(int64_t n) {
    static const std::map<std::string, std::string> ords = {
        { "one", "first" }, { "two", "second" }, { "three", "third" }, { "four", "fourth" }, { "five", "fifth" },
        { "six", "sixth" }, { "seven", "seventh" }, { "eight", "eighth" }, { "nine", "ninth" }, { "ten", "tenth" },
        { "eleven", "eleventh" }, { "twelve", "twelfth" },
    };
    std::string s = num_cardinal(n);
    size_t pos = s.find_last_of(" -");
    std::string last = pos == std::string::npos ? s : s.substr(pos + 1);
    std::string pre  = pos == std::string::npos ? "" : s.substr(0, pos + 1);
    auto it = ords.find(last);
    if (it != ords.end()) {
        last = it->second;
    } else {
        if (last.back() == 'y') {
            last = last.substr(0, last.size() - 1) + "ie";
        }
        last += "th";
    }
    return pre + last;
}

static std::string num_year(int64_t n) {
    const int64_t hi = n / 100;
    const int64_t lo = n % 100;
    if (n < 0 || hi == 0 || (hi % 10 == 0 && lo < 10) || hi >= 100) {
        return num_cardinal(n);
    }
    std::string low = lo == 0 ? "hundred" : lo < 10 ? "oh-" + num_cardinal(lo) : num_cardinal(lo);
    return num_cardinal(hi) + " " + low;
}

// digits to int64, false if empty, not all digits or too long
static bool parse_digits(const std::string & s, int64_t & v) {
    if (s.empty() || s.size() > 18 || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        return false;
    }
    v = std::stoll(s);
    return true;
}

// cardinal words, numbers too long for int64 are read digit by digit
static std::string num_cardinal_str(const std::string & s) {
    int64_t v;
    if (parse_digits(s, v)) {
        return num_cardinal(v);
    }
    std::string out;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            out += (out.empty() ? "" : " ") + std::string(NUM_ONES[c - '0']);
        }
    }
    return out;
}

static std::string num_ordinal_str(const std::string & s) {
    int64_t v;
    return parse_digits(s, v) ? num_ordinal(v) : num_cardinal_str(s);
}

// num2words(float(word)); word is [0-9]*.[0-9]*
static std::string num_float(const ustr & word) {
    const size_t dot = word.find('.');
    std::string ip = to_u8(word.substr(0, dot));
    std::string fp = dot == ustr::npos ? "" : to_u8(word.substr(dot + 1));
    while (!fp.empty() && fp.back() == '0') {
        fp.pop_back();
    }
    std::string out = ip.empty() ? std::string("zero") : num_cardinal_str(ip);
    if (!fp.empty()) {
        out += " point";
        for (char c : fp) {
            out += std::string(" ") + NUM_ONES[c - '0'];
        }
    }
    return out;
}

// re.split(r'[^a-z]+', s)
static std::vector<ustr> split_az(const std::string & s) {
    std::vector<ustr> out(1);
    bool in_sep = false;
    for (char c : s) {
        if (c >= 'a' && c <= 'z') {
            if (in_sep) {
                out.emplace_back();
                in_sep = false;
            }
            out.back() += (char32_t) c;
        } else {
            in_sep = true;
        }
    }
    if (in_sep) {
        out.emplace_back();
    }
    return out;
}

//
// tokens
//

struct mtoken {
    ustr text;
    std::string tag;
    ustr whitespace;
    ps_t phonemes;
    bool is_head = true;
    ps_t alias;
    stress_t stress;
    char32_t currency = 0;
    ustr num_flags;
    bool prespace = false;
};

struct token_ctx {
    std::optional<bool> future_vowel;
    bool future_to = false;
};

static const std::map<char32_t, std::pair<const char *, const char *>> CURRENCIES = {
    { U'$', { "dollar", "cent" } }, { U'£', { "pound", "pence" } }, { U'€', { "euro", "cent" } },
};

static mtoken merge_tokens(const std::vector<mtoken> & tks, const ps_t & unk) {
    mtoken m;
    for (size_t i = 0; i + 1 < tks.size(); i++) {
        m.text += tks[i].text + tks[i].whitespace;
    }
    m.text += tks.back().text;
    // tag of the token with most "weight", upper case letters count twice
    int best = -1;
    for (const auto & tk : tks) {
        int w = 0;
        for (char32_t c : tk.text) {
            w += c == lower_c(c) ? 1 : 2;
        }
        if (w > best) {
            best = w;
            m.tag = tk.tag;
        }
    }
    m.whitespace = tks.back().whitespace;
    if (unk) {
        ustr ps;
        for (const auto & tk : tks) {
            if (tk.prespace && !ps.empty() && !is_space_c(ps.back()) && tk.phonemes && !tk.phonemes->empty()) {
                ps += U' ';
            }
            ps += tk.phonemes ? *tk.phonemes : *unk;
        }
        m.phonemes = ps;
    }
    m.is_head = tks[0].is_head;
    std::set<double> stresses;
    for (const auto & tk : tks) {
        if (tk.stress) {
            stresses.insert(*tk.stress);
        }
        m.currency = std::max(m.currency, tk.currency);
        for (char32_t c : tk.num_flags) {
            if (!has(m.num_flags, c)) {
                m.num_flags += c;
            }
        }
    }
    std::sort(m.num_flags.begin(), m.num_flags.end());
    if (stresses.size() == 1) {
        m.stress = *stresses.begin();
    }
    m.prespace = tks[0].prespace;
    return m;
}

// regex.findall of misaki's SUBTOKEN_REGEX, alternatives tried in order at each position
static std::vector<ustr> subtokenize(const ustr & w) {
    auto is_apos = [](char32_t c) { return c == U'\'' || c == U'‘' || c == U'’'; };
    std::vector<ustr> out;
    const size_t n = w.size();
    size_t p = 0;
    while (p < n) {
        size_t e = p;
        // ^['\u2018\u2019]+
        if (p == 0 && is_apos(w[0])) {
            while (e < n && is_apos(w[e])) {
                e++;
            }
        }
        // \p{Lu}(?=\p{Lu}\p{Ll})
        if (e == p && is_upper(w[p]) && p + 2 < n && is_upper(w[p + 1]) && is_lower_letter(w[p + 2])) {
            e = p + 1;
        }
        // (?:^-)?(?:\d?[,.]?\d)+
        if (e == p) {
            size_t q = p;
            if (p == 0 && w[0] == U'-') {
                q++;
            }
            size_t r = q;
            auto sep = [&](size_t i) { return i < n && (w[i] == U',' || w[i] == U'.'); };
            auto dig = [&](size_t i) { return i < n && is_digit_c(w[i]); };
            while (true) {
                if (dig(r) && sep(r + 1) && dig(r + 2)) { r += 3; }
                else if (dig(r) && dig(r + 1))          { r += 2; }
                else if (sep(r) && dig(r + 1))          { r += 2; }
                else if (dig(r))                        { r += 1; }
                else                                    { break; }
            }
            if (r > q) {
                e = r;
            }
        }
        // [-_]+
        if (e == p && (w[p] == U'-' || w[p] == U'_')) {
            while (e < n && (w[e] == U'-' || w[e] == U'_')) {
                e++;
            }
        }
        // ['\u2018\u2019]{2,}
        if (e == p && is_apos(w[p]) && p + 1 < n && is_apos(w[p + 1])) {
            while (e < n && is_apos(w[e])) {
                e++;
            }
        }
        // \p{L}*?(?:['\u2018\u2019]\p{L})*?\p{Ll}(?=\p{Lu}): shortest letter run ending lower case before an upper case letter
        if (e == p && is_letter(w[p])) {
            for (size_t k = p; k + 1 < n && is_letter(w[k]); k++) {
                if (is_lower_letter(w[k]) && is_upper(w[k + 1])) {
                    e = k + 1;
                    break;
                }
            }
        }
        // \p{L}+(?:['\u2018\u2019]\p{L})*
        if (e == p && is_letter(w[p])) {
            while (e < n && is_letter(w[e])) {
                e++;
            }
            while (e + 1 < n && is_apos(w[e]) && is_letter(w[e + 1])) {
                e += 2;
            }
        }
        // [^-_\p{L}'\u2018\u2019\d]
        if (e == p && !(w[p] == U'-' || w[p] == U'_' || is_letter(w[p]) || is_apos(w[p]) || is_digit_c(w[p]))) {
            e = p + 1;
        }
        // ['\u2018\u2019]+$
        if (e == p && is_apos(w[p])) {
            size_t q = p;
            while (q < n && is_apos(w[q])) {
                q++;
            }
            if (q == n) {
                e = q;
            }
        }
        if (e == p) {
            p++; // no match here, findall moves on
            continue;
        }
        out.push_back(w.substr(p, e - p));
        p = e;
    }
    return out;
}

//
// lexicon
//

struct lex_entry {
    ps_t def;
    std::map<std::string, ps_t> tags; // heteronym variants
};

struct g2p_en::impl {
    bool british;
    std::unordered_map<ustr, lex_entry> golds;
    std::unordered_map<ustr, ustr> silvers;

    static std::string parent_tag(const std::string & tag) {
        if (tag.rfind("VB", 0) == 0) return "VERB";
        if (tag.rfind("NN", 0) == 0) return "NOUN";
        if (tag.rfind("ADV", 0) == 0 || tag.rfind("RB", 0) == 0) return "ADV";
        if (tag.rfind("ADJ", 0) == 0 || tag.rfind("JJ", 0) == 0) return "ADJ";
        return tag;
    }

    ps_t gold_default(const char32_t * w) const {
        auto it = golds.find(w);
        return it == golds.end() ? ps_t() : it->second.def;
    }

    ps_t get_NNP(const ustr & word) const {
        ustr ps;
        for (char32_t c : word) {
            if (!is_letter(c)) {
                continue;
            }
            auto it = golds.find(ustr(1, upper_c(c)));
            if (it == golds.end() || !it->second.def) {
                return {};
            }
            ps += *it->second.def;
        }
        ps = apply_stress(ps, 0.0);
        const size_t pos = ps.rfind(SECONDARY);
        if (pos != ustr::npos) {
            ps[pos] = PRIMARY;
        }
        return ps;
    }

    ps_t get_special_case(const ustr & word, const std::string & tag, stress_t stress, const token_ctx & ctx) const {
        static const std::map<ustr, const char32_t *> add_symbols = { { U".", U"dot" }, { U"/", U"slash" } };
        static const std::map<ustr, const char32_t *> symbols = { { U"%", U"percent" }, { U"&", U"and" }, { U"+", U"plus" }, { U"@", U"at" } };
        if (tag == "ADD" && add_symbols.count(word)) {
            return lookup(add_symbols.at(word), "", -0.5, &ctx);
        }
        if (symbols.count(word)) {
            return lookup(symbols.at(word), "", {}, &ctx);
        }
        {
            // abbreviations like U.S.A.: spell the letters
            ustr stripped = word;
            while (!stripped.empty() && stripped.front() == U'.') stripped.erase(0, 1);
            while (!stripped.empty() && stripped.back() == U'.') stripped.pop_back();
            size_t longest = 0;
            size_t cur = 0;
            for (char32_t c : word) {
                cur = c == U'.' ? 0 : cur + 1;
                longest = std::max(longest, cur);
            }
            if (has(stripped, U'.') && is_alpha(remove_c(word, U'.')) && longest < 3) {
                return get_NNP(word);
            }
        }
        if (word == U"a" || word == U"A") {
            return tag == "DT" ? ustr(U"ɐ") : ustr(U"ˈA");
        }
        if (word == U"am" || word == U"Am" || word == U"AM") {
            if (tag.rfind("NN", 0) == 0) {
                return get_NNP(word);
            }
            if (!ctx.future_vowel || word != U"am" || (stress && *stress > 0)) {
                return gold_default(U"am");
            }
            return ustr(U"ɐm");
        }
        if (word == U"an" || word == U"An" || word == U"AN") {
            if (word == U"AN" && tag.rfind("NN", 0) == 0) {
                return get_NNP(word);
            }
            return ustr(U"ɐn");
        }
        if (word == U"I" && tag == "PRP") {
            return ustr(U"ˌI");
        }
        if ((word == U"by" || word == U"By" || word == U"BY") && parent_tag(tag) == "ADV") {
            return ustr(U"bˈI");
        }
        if (word == U"to" || word == U"To" || (word == U"TO" && (tag == "TO" || tag == "IN"))) {
            if (!ctx.future_vowel) {
                return gold_default(U"to");
            }
            return *ctx.future_vowel ? ustr(U"tʊ") : ustr(U"tə");
        }
        if (word == U"in" || word == U"In" || (word == U"IN" && tag != "NNP")) {
            return (!ctx.future_vowel || tag != "IN" ? ustr(1, PRIMARY) : ustr()) + U"ɪn";
        }
        if (word == U"the" || word == U"The" || (word == U"THE" && tag == "DT")) {
            return ctx.future_vowel && *ctx.future_vowel ? ustr(U"ði") : ustr(U"ðə");
        }
        if (tag == "IN" && (lower(word) == U"vs" || lower(word) == U"vs.")) {
            return lookup(U"versus", "", {}, &ctx);
        }
        if (word == U"used" || word == U"Used" || word == U"USED") {
            auto it = golds.find(U"used");
            if (it != golds.end()) {
                if ((tag == "VBD" || tag == "JJ") && ctx.future_to && it->second.tags.count("VBD")) {
                    return it->second.tags.at("VBD");
                }
                return it->second.def;
            }
        }
        return {};
    }

    bool is_known(const ustr & word, const std::string & /*tag*/) const {
        static const ustr symbols = U"%&+@";
        if (golds.count(word) || (word.size() == 1 && has(symbols, word[0])) || silvers.count(word)) {
            return true;
        }
        auto lex_ord = [](char32_t c) { return c == U'\'' || c == U'-' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
        if (!is_alpha(word) || !std::all_of(word.begin(), word.end(), lex_ord)) {
            return false;
        }
        if (word.size() == 1) {
            return true;
        }
        if (word == upper(word) && golds.count(lower(word))) {
            return true;
        }
        return word.substr(1) == upper(word.substr(1));
    }

    ps_t lookup(ustr word, std::string tag, stress_t stress, const token_ctx * ctx) const {
        bool is_nnp = false;
        if (word == upper(word) && !golds.count(word)) {
            word = lower(word);
            is_nnp = tag == "NNP";
        }
        ps_t ps;
        auto g = golds.find(word);
        if (g != golds.end()) {
            const auto & e = g->second;
            if (e.tags.empty()) {
                ps = e.def;
            } else {
                if (ctx && !ctx->future_vowel && e.tags.count("None")) {
                    tag = "None";
                } else if (tag == "VBN?") {
                    // after a form of "be": only an explicit participle variant, adjectives keep the default
                    tag = "VBN";
                } else if (!e.tags.count(tag)) {
                    tag = parent_tag(tag);
                }
                auto t = e.tags.find(tag);
                ps = t != e.tags.end() ? t->second : e.def;
            }
        } else if (!is_nnp) {
            auto s = silvers.find(word);
            if (s != silvers.end()) {
                ps = s->second;
            }
        }
        if (!ps || (is_nnp && !has(*ps, PRIMARY))) {
            ps_t n = get_NNP(word);
            if (n) {
                return n;
            }
            return {};
        }
        return apply_stress(*ps, stress);
    }

    ps_t suffix_s(const ps_t & stem) const {
        if (!stem || stem->empty()) {
            return {};
        }
        const char32_t c = stem->back();
        if (has(U"ptkfθ", c)) {
            return *stem + U"s";
        }
        if (has(U"szʃʒʧʤ", c)) {
            return *stem + (british ? U"ɪ" : U"ᵻ") + U"z";
        }
        return *stem + U"z";
    }

    ps_t stem_s(const ustr & word, const std::string & tag, stress_t stress, const token_ctx * ctx) const {
        if (word.size() < 3 || !ends_with(word, U"s")) {
            return {};
        }
        ustr stem;
        if (!ends_with(word, U"ss") && is_known(head(word, 1), tag)) {
            stem = head(word, 1);
        } else if ((ends_with(word, U"'s") || (word.size() > 4 && ends_with(word, U"es") && !ends_with(word, U"ies"))) && is_known(head(word, 2), tag)) {
            stem = head(word, 2);
        } else if (word.size() > 4 && ends_with(word, U"ies") && is_known(head(word, 3) + U"y", tag)) {
            stem = head(word, 3) + U"y";
        } else {
            return {};
        }
        return suffix_s(lookup(stem, tag, stress, ctx));
    }

    ps_t suffix_ed(const ps_t & stem) const {
        if (!stem || stem->empty()) {
            return {};
        }
        const ustr & s = *stem;
        const char32_t c = s.back();
        if (has(U"pkfθʃsʧ", c)) {
            return s + U"t";
        }
        if (c == U'd') {
            return s + (british ? U"ɪ" : U"ᵻ") + U"d";
        }
        if (c != U't') {
            return s + U"d";
        }
        if (british || s.size() < 2) {
            return s + U"ɪd";
        }
        if (has(US_TAUS, s[s.size() - 2])) {
            return head(s, 1) + U"ɾᵻd";
        }
        return s + U"ᵻd";
    }

    ps_t stem_ed(const ustr & word, const std::string & tag, stress_t stress, const token_ctx * ctx) const {
        if (word.size() < 4 || !ends_with(word, U"d")) {
            return {};
        }
        ustr stem;
        if (!ends_with(word, U"dd") && is_known(head(word, 1), tag)) {
            stem = head(word, 1);
        } else if (word.size() > 4 && ends_with(word, U"ed") && !ends_with(word, U"eed") && is_known(head(word, 2), tag)) {
            stem = head(word, 2);
        } else {
            return {};
        }
        return suffix_ed(lookup(stem, tag, stress, ctx));
    }

    ps_t suffix_ing(const ps_t & stem) const {
        if (!stem || stem->empty()) {
            return {};
        }
        const ustr & s = *stem;
        if (british) {
            if (has(U"əː", s.back())) {
                return {};
            }
        } else if (s.size() > 1 && s.back() == U't' && has(US_TAUS, s[s.size() - 2])) {
            return head(s, 1) + U"ɾɪŋ";
        }
        return s + U"ɪŋ";
    }

    ps_t stem_ing(const ustr & word, const std::string & tag, stress_t stress, const token_ctx * ctx) const {
        if (word.size() < 5 || !ends_with(word, U"ing")) {
            return {};
        }
        ustr stem;
        // ([bcdgklmnprstvxz])\1ing$|cking$
        auto doubled = [&]() {
            if (ends_with(word, U"cking")) {
                return true;
            }
            const size_t n = word.size();
            return n >= 5 && word[n - 4] == word[n - 5] && has(U"bcdgklmnprstvxz", word[n - 4]);
        };
        if (word.size() > 5 && is_known(head(word, 3), tag)) {
            stem = head(word, 3);
        } else if (is_known(head(word, 3) + U"e", tag)) {
            stem = head(word, 3) + U"e";
        } else if (word.size() > 5 && doubled() && is_known(head(word, 4), tag)) {
            stem = head(word, 4);
        } else {
            return {};
        }
        return suffix_ing(lookup(stem, tag, stress, ctx));
    }

    ps_t get_word(ustr word, const std::string & tag, stress_t stress, const token_ctx & ctx) const {
        ps_t sc = get_special_case(word, tag, stress, ctx);
        if (sc) {
            return sc;
        }
        const ustr wl = lower(word);
        if (word.size() > 1 && is_alpha(remove_c(word, U'\'')) && word != wl && (tag != "NNP" || word.size() > 7) &&
                !golds.count(word) && !silvers.count(word) && (word == upper(word) || word.substr(1) == lower(word.substr(1))) &&
                (golds.count(wl) || silvers.count(wl) || stem_s(wl, tag, stress, &ctx) || stem_ed(wl, tag, stress, &ctx) || stem_ing(wl, tag, stress, &ctx))) {
            word = wl;
        }
        if (is_known(word, tag)) {
            return lookup(word, tag, stress, &ctx);
        }
        if (ends_with(word, U"s'") && is_known(head(word, 2) + U"'s", tag)) {
            return lookup(head(word, 2) + U"'s", tag, stress, &ctx);
        }
        if (ends_with(word, U"'") && is_known(head(word, 1), tag)) {
            return lookup(head(word, 1), tag, stress, &ctx);
        }
        if (ps_t s = stem_s(word, tag, stress, &ctx)) {
            return s;
        }
        if (ps_t s = stem_ed(word, tag, stress, &ctx)) {
            return s;
        }
        if (ps_t s = stem_ing(word, tag, stress ? stress : 0.5, &ctx)) {
            return s;
        }
        return {};
    }

    static bool is_currency(const ustr & word) {
        const size_t dots = std::count(word.begin(), word.end(), U'.');
        if (dots == 0) {
            return true;
        }
        if (dots > 1) {
            return false;
        }
        return word.size() - word.find(U'.') - 1 < 3;
    }

    static bool is_number(ustr word, bool is_head) {
        if (std::none_of(word.begin(), word.end(), is_digit_c)) {
            return false;
        }
        for (const char32_t * s : { U"ing", U"'d", U"ed", U"'s", U"st", U"nd", U"rd", U"th", U"s" }) {
            if (ends_with(word, s)) {
                word = head(word, std::char_traits<char32_t>::length(s));
                break;
            }
        }
        for (size_t i = 0; i < word.size(); i++) {
            const char32_t c = word[i];
            if (!(is_digit_c(c) || c == U',' || c == U'.' || (is_head && i == 0 && c == U'-'))) {
                return false;
            }
        }
        return true;
    }

    ps_t get_number(ustr word, char32_t currency, bool is_head, const ustr & num_flags) const {
        // trailing [a-z']+
        size_t k = word.size();
        while (k > 0 && ((word[k - 1] >= 'a' && word[k - 1] <= 'z') || word[k - 1] == U'\'')) {
            k--;
        }
        const ustr suffix = word.substr(k);
        word = word.substr(0, k);
        const bool ordinal_suffix = suffix == U"st" || suffix == U"nd" || suffix == U"rd" || suffix == U"th";
        std::vector<ps_t> result;
        if (starts_with(word, U"-")) {
            result.push_back(lookup(U"minus", "", {}, nullptr));
            word = word.substr(1);
        }
        auto extend_words = [&](const std::string & words) {
            for (const auto & w : split_az(words)) {
                if (w != U"and" || has(num_flags, U'&')) {
                    result.push_back(lookup(w, "", w == U"point" ? stress_t(-2) : stress_t(), nullptr));
                } else if (has(num_flags, U'n') && !result.empty() && result.back()) {
                    *result.back() += U"ən";
                }
            }
        };
        auto extend_num = [&](const ustr & num) {
            extend_words(num_cardinal_str(to_u8(num)));
        };
        const bool digits = is_digits(word);
        const bool has_dot = has(word, U'.');
        if (word.empty()) {
            return {};
        }
        if (digits && ordinal_suffix) {
            extend_words(num_ordinal_str(to_u8(word)));
        } else if (result.empty() && word.size() == 4 && !CURRENCIES.count(currency) && digits) {
            int64_t year = 0;
            parse_digits(to_u8(word), year);
            extend_words(num_year(year));
        } else if (!is_head && !has_dot) {
            const ustr num = remove_c(word, U',');
            if (num.empty()) {
                return {};
            }
            if (num[0] == U'0' || num.size() > 3) {
                for (char32_t c : num) {
                    extend_num(ustr(1, c));
                }
            } else if (num.size() == 3 && !ends_with(num, U"00")) {
                extend_num(num.substr(0, 1));
                if (num[1] == U'0') {
                    result.push_back(lookup(U"O", "", -2, nullptr));
                    extend_num(num.substr(2, 1));
                } else {
                    extend_num(num.substr(1));
                }
            } else {
                extend_num(num);
            }
        } else if (std::count(word.begin(), word.end(), U'.') > 1 || !is_head) {
            bool first = true;
            ustr rest = remove_c(word, U',');
            size_t pos = 0;
            while (pos <= rest.size()) {
                size_t e = rest.find(U'.', pos);
                if (e == ustr::npos) {
                    e = rest.size();
                }
                const ustr num = rest.substr(pos, e - pos);
                pos = e + 1;
                if (num.empty()) {
                    first = false;
                    continue;
                }
                const bool spell = num[0] == U'0' || (num.size() != 2 && std::any_of(num.begin() + 1, num.end(), [](char32_t c) { return c != U'0'; }));
                if (spell) {
                    for (char32_t c : num) {
                        extend_num(ustr(1, c));
                    }
                } else {
                    extend_num(num);
                }
                first = false;
            }
            (void) first;
        } else if (CURRENCIES.count(currency) && is_currency(word) && remove_c(word, U',').size() <= 18) {
            const auto & units = CURRENCIES.at(currency);
            const ustr w = remove_c(word, U',');
            const size_t dot = w.find(U'.');
            std::vector<std::pair<int64_t, std::string>> pairs;
            const ustr major = w.substr(0, dot);
            int64_t v_major = 0;
            parse_digits(to_u8(major), v_major);
            pairs.push_back({ v_major, units.first });
            if (dot != ustr::npos) {
                const ustr minor = w.substr(dot + 1);
                int64_t v_minor = 0;
                parse_digits(to_u8(minor), v_minor);
                pairs.push_back({ v_minor, units.second });
            }
            if (pairs.size() > 1) {
                if (pairs[1].first == 0) {
                    pairs.resize(1);
                } else if (pairs[0].first == 0) {
                    pairs.erase(pairs.begin());
                }
            }
            for (size_t i = 0; i < pairs.size(); i++) {
                if (i > 0) {
                    result.push_back(lookup(U"and", "", {}, nullptr));
                }
                extend_words(num_cardinal(pairs[i].first));
                const ustr unit = to_u32(pairs[i].second);
                if (std::abs(pairs[i].first) != 1 && unit != U"pence") {
                    result.push_back(stem_s(unit + U"s", "", {}, nullptr));
                } else {
                    result.push_back(lookup(unit, "", {}, nullptr));
                }
            }
        } else {
            std::string words;
            if (digits) {
                words = num_cardinal_str(to_u8(word));
            } else if (!has_dot) {
                const std::string v = to_u8(remove_c(word, U','));
                words = ordinal_suffix ? num_ordinal_str(v) : num_cardinal_str(v);
            } else {
                const ustr w = remove_c(word, U',');
                if (w[0] == U'.') {
                    words = "point";
                    for (char32_t c : w.substr(1)) {
                        if (is_digit_c(c)) {
                            words += std::string(" ") + NUM_ONES[c - '0'];
                        }
                    }
                } else {
                    words = num_float(w);
                }
            }
            extend_words(words);
        }
        if (result.empty()) {
            return {};
        }
        ustr ps;
        for (size_t i = 0; i < result.size(); i++) {
            if (!result[i]) {
                return {};
            }
            ps += (i ? U" " : U"") + *result[i];
        }
        if (suffix == U"s" || suffix == U"'s") {
            return suffix_s(ps);
        }
        if (suffix == U"ed" || suffix == U"'d") {
            return suffix_ed(ps);
        }
        if (suffix == U"ing") {
            return suffix_ing(ps);
        }
        return ps;
    }

    ustr append_currency(const ustr & ps, char32_t currency) const {
        auto it = CURRENCIES.find(currency);
        if (it == CURRENCIES.end()) {
            return ps;
        }
        ps_t c = stem_s(to_u32(it->second.first) + U"s", "", {}, nullptr);
        return c ? ps + U" " + *c : ps;
    }

    ps_t lexicon(const mtoken & tk, const token_ctx & ctx) const {
        ustr word = tk.alias ? *tk.alias : tk.text;
        word = replace_all(replace_all(word, U"‘", U"'"), U"’", U"'");
        const stress_t stress = word == lower(word) ? stress_t() : stress_t(word == upper(word) ? 2.0 : 0.5);
        ps_t ps = get_word(word, tk.tag, stress, ctx);
        if (ps) {
            return apply_stress(append_currency(*ps, tk.currency), tk.stress);
        }
        if (is_number(word, tk.is_head)) {
            ps = get_number(word, tk.currency, tk.is_head, tk.num_flags);
            return ps ? ps_t(apply_stress(*ps, tk.stress)) : ps_t();
        }
        return {};
    }

    //
    // tokenizer, a rule-based stand-in for the spaCy English tokenizer and tagger
    //

    static std::string tag_of(const ustr & t, const ustr & next, bool opening) {
        if (t == U"." || t == U"!" || t == U"?" || t == U"?!" || t == U"!?") return ".";
        if (t == U",") return ",";
        if (t == U";" || t == U":" || t == U"..." || t == U"…" || t == U"--" || t == U"—" || t == U"–") return ":";
        if (t == U"-") return "HYPH";
        if (t == U"\"") return opening ? "``" : "''";
        // a straight apostrophe after a word is a possessive, not a quote
        if (t == U"'") return opening ? "``" : "POS";
        if (t == U"“" || t == U"‘") return "``";
        if (t == U"”" || t == U"’") return "''";
        if (t == U"(" || t == U"[" || t == U"{") return "-LRB-";
        if (t == U")" || t == U"]" || t == U"}") return "-RRB-";
        if (t == U"$" || t == U"£" || t == U"€") return "$";
        if (t == U"#") return "#";
        if (!t.empty() && (is_digit_c(t[0]) || (t.size() > 1 && (t[0] == U'.' || t[0] == U'-') && is_digit_c(t[1])))) {
            const bool num = std::all_of(t.begin(), t.end(), [](char32_t c) { return is_digit_c(c) || c == U',' || c == U'.' || c == U':' || c == U'-'; });
            return num ? "CD" : "";
        }
        const ustr tl = lower(t);
        if (tl == U"hundred" || tl == U"thousand" || tl == U"million" || tl == U"billion" || tl == U"trillion") return "CD";
        if (tl == U"a" || tl == U"an" || tl == U"the") return "DT";
        if (t == U"I") return "PRP";
        if (tl == U"to") return "TO";
        if (tl == U"in") return "IN";
        if (tl == U"vs" || tl == U"vs.") return "IN";
        if (tl == U"am") return "VBP";
        if (tl == U"used") return lower(next) == U"to" ? "VBD" : "VBN";
        if (t.size() > 1 && is_alpha(t) && t == upper(t)) return "NNP";
        return "";
    }

    // coarse part of speech from the previous word, enough to pick most heteronym variants of the lexicon
    static std::string context_tag(const ustr & prev) {
        static const std::set<ustr> verb_before = {
            U"to", U"can", U"could", U"will", U"would", U"shall", U"should", U"may", U"might", U"must",
            U"do", U"does", U"did", U"not", U"n't", U"n’t", U"'ll", U"’ll", U"'d", U"’d",
        };
        static const std::set<ustr> pronouns  = { U"i", U"you", U"we", U"they", U"he", U"she", U"it" };
        static const std::set<ustr> be_forms  = { U"was", U"were", U"is", U"are", U"be", U"been", U"being", U"am", U"'re", U"’re" };
        static const std::set<ustr> noun_before = {
            U"the", U"a", U"an", U"this", U"these", U"those", U"his", U"her", U"my", U"your", U"our", U"their", U"its",
            U"some", U"any", U"no", U"every", U"each",
        };
        if (verb_before.count(prev)) return "VB";
        if (pronouns.count(prev))    return "VBP";
        // participle or adjective, see lookup()
        if (be_forms.count(prev))    return "VBN?";
        if (noun_before.count(prev)) return "NN";
        return "";
    }

    // "that" is DT (stressed in the lexicon) as a demonstrative, else IN or WDT like spaCy tags it
    static std::string that_tag(const ustr & prev, const ustr & next) {
        static const std::set<ustr> clause_start = { U"", U".", U"!", U"?", U"\"", U"\u201c", U"(", U"and", U"but", U"or", U"so" };
        static const std::set<ustr> subjects = {
            U"i", U"you", U"we", U"they", U"he", U"she", U"it", U"there", U"the", U"a", U"an", U"this", U"these", U"those",
            U"nobody", U"everybody", U"everyone", U"someone", U"something", U"everything", U"nothing",
        };
        static const std::set<ustr> aux = {
            U"is", U"are", U"was", U"were", U"will", U"would", U"can", U"could", U"should", U"may", U"might", U"must",
            U"has", U"have", U"had", U"does", U"did",
        };
        const ustr n = lower(next);
        auto ends = [&](const ustr & e) { return n.size() > e.size() + 1 && n.compare(n.size() - e.size(), e.size(), e) == 0; };
        // conjunction before a clause, "not that"
        if (subjects.count(n) || lower(prev) == U"not") return "IN";
        if (clause_start.count(lower(prev)) || n == U"'s" || n == U"\u2019s") return "DT";
        // relative before a verb: aux, past tense, 3rd person -s (a demonstrative takes a singular noun)
        if (aux.count(n) || ends(U"ed") || (ends(U"s") && !ends(U"ss") && !ends(U"us") && !ends(U"is"))) return "WDT";
        // before a name
        if (!next.empty() && next[0] != lower(next)[0]) return "IN";
        return "DT";
    }

    static std::vector<mtoken> tokenize(const ustr & text) {
        static const std::set<ustr> abbrevs = {
            U"mr", U"mrs", U"ms", U"dr", U"prof", U"st", U"jr", U"sr", U"vs", U"etc", U"e.g", U"i.e", U"inc", U"ltd",
            U"jan", U"feb", U"apr", U"jul", U"aug", U"sep", U"sept", U"oct", U"nov", U"dec", U"approx",
        };
        static const std::vector<ustr> contractions = { U"n't", U"n’t", U"'s", U"’s", U"'m", U"’m", U"'d", U"’d", U"'ll", U"’ll", U"'re", U"’re", U"'ve", U"’ve" };
        static const ustr prefix_chars = U"\"'([{“‘$£€#¿¡*";
        static const ustr suffix_chars = U",;:!?)]}\"”’'%";

        struct piece {
            ustr text;
            bool opening;
        };
        std::vector<std::pair<std::vector<piece>, bool>> chunks; // pieces, whitespace after
        size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && is_space_c(text[i])) {
                i++;
            }
            size_t e = i;
            while (e < text.size() && !is_space_c(text[e])) {
                e++;
            }
            if (e == i) {
                break;
            }
            ustr s = text.substr(i, e - i);
            const bool ws = e < text.size();
            i = e;

            std::vector<piece> pre;
            std::vector<piece> post;
            bool changed = true;
            while (changed && !s.empty()) {
                changed = false;
                if (has(prefix_chars, s[0]) && s.size() > 1) {
                    pre.push_back({ s.substr(0, 1), true });
                    s = s.substr(1);
                    changed = true;
                    continue;
                }
                for (const ustr & el : { ustr(U"..."), ustr(U"…"), ustr(U"--"), ustr(U"—") }) {
                    if (s.size() > el.size() && ends_with(s, el)) {
                        post.insert(post.begin(), { el, false });
                        s = head(s, el.size());
                        changed = true;
                        break;
                    }
                }
                if (changed) {
                    continue;
                }
                if (s.size() > 1 && has(suffix_chars, s.back())) {
                    // keep the apostrophe of a plural possessive with its word
                    post.insert(post.begin(), { s.substr(s.size() - 1), false });
                    s = head(s, 1);
                    changed = true;
                    continue;
                }
                if (s.size() > 1 && s.back() == U'.') {
                    const ustr body = lower(head(s, 1));
                    // dotted letter groups like U.S.A or Ph.D, as in Lexicon.get_special_case()
                    size_t longest = 0;
                    size_t cur_len = 0;
                    for (char32_t c : body) {
                        cur_len = c == U'.' ? 0 : cur_len + 1;
                        longest = std::max(longest, cur_len);
                    }
                    const bool dotted = has(body, U'.') && is_alpha(remove_c(body, U'.')) && longest < 3;
                    const bool abbrev = dotted || abbrevs.count(body);
                    if (!abbrev) {
                        post.insert(post.begin(), { ustr(U"."), false });
                        s = head(s, 1);
                        changed = true;
                        continue;
                    }
                }
                for (const ustr & c : contractions) {
                    if (s.size() > c.size() && ends_with(lower(s), c)) {
                        post.insert(post.begin(), { s.substr(s.size() - c.size()), false });
                        s = head(s, c.size());
                        changed = true;
                        break;
                    }
                }
            }
            // infixes: dashes, hyphens between letters
            std::vector<piece> mid;
            ustr cur;
            for (size_t k = 0; k < s.size(); k++) {
                const bool dash2 = s[k] == U'-' && k + 1 < s.size() && s[k + 1] == U'-';
                const bool emdash = s[k] == U'—';
                const bool hyph = s[k] == U'-' && k > 0 && k + 1 < s.size() && is_letter(s[k - 1]) && is_letter(s[k + 1]);
                if (dash2 || emdash || hyph) {
                    if (!cur.empty()) {
                        mid.push_back({ cur, false });
                        cur.clear();
                    }
                    mid.push_back({ dash2 ? ustr(U"--") : ustr(1, s[k]), false });
                    if (dash2) {
                        k++;
                    }
                } else {
                    cur += s[k];
                }
            }
            if (!cur.empty()) {
                mid.push_back({ cur, false });
            }
            std::vector<piece> all = pre;
            all.insert(all.end(), mid.begin(), mid.end());
            all.insert(all.end(), post.begin(), post.end());
            chunks.push_back({ all, ws });
        }

        std::vector<mtoken> tokens;
        std::vector<std::pair<ustr, bool>> flat;
        std::vector<bool> ws_after;
        for (const auto & c : chunks) {
            for (size_t k = 0; k < c.first.size(); k++) {
                flat.push_back({ c.first[k].text, c.first[k].opening });
                ws_after.push_back(k + 1 == c.first.size() && c.second);
            }
        }
        for (size_t k = 0; k < flat.size(); k++) {
            mtoken tk;
            tk.text = flat[k].first;
            tk.whitespace = ws_after[k] ? U" " : U"";
            const ustr next = k + 1 < flat.size() ? flat[k + 1].first : ustr();
            // a straight quote opens if it starts a chunk
            const bool opening = flat[k].second || k == 0 || (k > 0 && ws_after[k - 1]);
            tk.tag = tag_of(tk.text, next, opening);
            if (lower(tk.text) == U"that") {
                tk.tag = that_tag(k > 0 ? flat[k - 1].first : ustr(), next);
            } else if (k > 0 && (tk.tag.empty() || (lower(tk.text) == U"used" && context_tag(lower(flat[k - 1].first)) == "VBN?"))) {
                tk.tag = context_tag(lower(flat[k - 1].first));
            }
            tokens.push_back(tk);
        }
        return tokens;
    }

    struct word_t {
        std::vector<mtoken> tks;
        bool is_list;
    };

    static std::vector<word_t> retokenize(const std::vector<mtoken> & tokens) {
        static const std::set<std::string> punct_tags = { ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP" };
        static const std::map<std::string, const char32_t *> punct_tag_phonemes = {
            { "-LRB-", U"(" }, { "-RRB-", U")" }, { "``", U"“" }, { "\"\"", U"”" }, { "''", U"”" },
        };
        std::vector<word_t> words;
        char32_t currency = 0;
        for (size_t i = 0; i < tokens.size(); i++) {
            const mtoken & token = tokens[i];
            std::vector<mtoken> tks;
            if (!token.alias && !token.phonemes) {
                for (const auto & t : subtokenize(token.text)) {
                    mtoken s = token;
                    s.text = t;
                    s.whitespace.clear();
                    s.is_head = true;
                    s.prespace = false;
                    tks.push_back(s);
                }
            } else {
                tks.push_back(token);
            }
            if (tks.empty()) {
                continue;
            }
            tks.back().whitespace = token.whitespace;
            for (size_t j = 0; j < tks.size(); j++) {
                mtoken & tk = tks[j];
                if (tk.alias || tk.phonemes) {
                    // nothing
                } else if (tk.tag == "$" && tk.text.size() == 1 && CURRENCIES.count(tk.text[0])) {
                    currency = tk.text[0];
                    tk.phonemes = ustr();
                } else if (tk.tag == ":" && (tk.text == U"-" || tk.text == U"–")) {
                    tk.phonemes = ustr(U"—");
                } else if (punct_tags.count(tk.tag) && !std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { char32_t l = lower_c(c); return l >= 'a' && l <= 'z'; })) {
                    auto it = punct_tag_phonemes.find(tk.tag);
                    if (it != punct_tag_phonemes.end()) {
                        tk.phonemes = ustr(it->second);
                    } else {
                        ustr p;
                        for (char32_t c : tk.text) {
                            if (has(PUNCTS, c)) {
                                p += c;
                            }
                        }
                        tk.phonemes = p;
                    }
                } else if (currency) {
                    if (tk.tag != "CD") {
                        currency = 0;
                    } else if (j + 1 == tks.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) {
                        tk.currency = currency;
                    }
                } else if (j > 0 && j + 1 < tks.size() && tk.text == U"2" && !tks[j - 1].text.empty() && !tks[j + 1].text.empty() &&
                           is_letter(tks[j - 1].text.back()) && is_letter(tks[j + 1].text[0])) {
                    tk.alias = ustr(U"to");
                }
                if (tk.alias || tk.phonemes) {
                    words.push_back({ { tk }, false });
                } else if (!words.empty() && words.back().is_list && words.back().tks.back().whitespace.empty()) {
                    tk.is_head = false;
                    words.back().tks.push_back(tk);
                } else {
                    words.push_back({ { tk }, tk.whitespace.empty() });
                }
            }
        }
        for (auto & w : words) {
            if (w.is_list && w.tks.size() == 1) {
                w.is_list = false;
            }
        }
        return words;
    }

    static token_ctx token_context(const token_ctx & ctx, const ps_t & ps, const mtoken & token) {
        token_ctx out;
        out.future_vowel = ctx.future_vowel;
        if (ps && !ps->empty()) {
            for (char32_t c : *ps) {
                if (has(VOWELS, c) || has(CONSONANTS, c) || has(NON_QUOTE_PUNCTS, c)) {
                    out.future_vowel = has(NON_QUOTE_PUNCTS, c) ? std::optional<bool>() : std::optional<bool>(has(VOWELS, c));
                    break;
                }
            }
        }
        out.future_to = token.text == U"to" || token.text == U"To" || (token.text == U"TO" && (token.tag == "TO" || token.tag == "IN"));
        return out;
    }

    static void resolve_tokens(std::vector<mtoken> & tokens) {
        ustr text;
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            text += tokens[i].text + tokens[i].whitespace;
        }
        text += tokens.back().text;
        std::set<int> kinds;
        for (char32_t c : text) {
            if (!has(SUBTOKEN_JUNKS, c)) {
                kinds.insert(is_letter(c) ? 0 : is_digit_c(c) ? 1 : 2);
            }
        }
        const bool prespace = has(text, U' ') || has(text, U'/') || kinds.size() > 1;
        for (size_t i = 0; i < tokens.size(); i++) {
            mtoken & tk = tokens[i];
            if (!tk.phonemes) {
                if (i + 1 == tokens.size() && tk.text.size() == 1 && has(NON_QUOTE_PUNCTS, tk.text[0])) {
                    tk.phonemes = tk.text;
                } else if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return has(SUBTOKEN_JUNKS, c); })) {
                    tk.phonemes = ustr();
                }
            } else if (i > 0) {
                tk.prespace = prespace;
            }
        }
        if (prespace) {
            return;
        }
        struct idx_t {
            bool primary;
            int weight;
            size_t i;
            bool operator<(const idx_t & o) const {
                return std::tie(primary, weight, i) < std::tie(o.primary, o.weight, o.i);
            }
        };
        std::vector<idx_t> indices;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i].phonemes && !tokens[i].phonemes->empty()) {
                indices.push_back({ has(*tokens[i].phonemes, PRIMARY), stress_weight(*tokens[i].phonemes), i });
            }
        }
        if (indices.size() == 2 && tokens[indices[0].i].text.size() == 1) {
            auto & ps = tokens[indices[1].i].phonemes;
            ps = apply_stress(*ps, -0.5);
            return;
        }
        size_t n_primary = 0;
        for (const auto & x : indices) {
            n_primary += x.primary;
        }
        if (indices.size() < 2 || n_primary <= (indices.size() + 1) / 2) {
            return;
        }
        std::sort(indices.begin(), indices.end());
        indices.resize(indices.size() / 2);
        for (const auto & x : indices) {
            tokens[x.i].phonemes = apply_stress(*tokens[x.i].phonemes, -0.5);
        }
    }

    // G2P.__call__ for one paragraph, returns the merged tokens
    std::vector<mtoken> g2p(const ustr & text, const fallback_t & fallback) const {
        std::vector<word_t> words = retokenize(tokenize(text));
        auto run_fallback = [&](const mtoken & tk) -> ps_t {
            std::string ps;
            if (fallback && fallback(to_u8(tk.text), ps) && !ps.empty()) {
                return to_u32(ps);
            }
            return {};
        };
        token_ctx ctx;
        for (size_t wi = words.size(); wi-- > 0; ) {
            word_t & w = words[wi];
            if (!w.is_list) {
                mtoken & tk = w.tks[0];
                if (!tk.phonemes) {
                    tk.phonemes = lexicon(tk, ctx);
                }
                if (!tk.phonemes) {
                    tk.phonemes = run_fallback(tk);
                }
                ctx = token_context(ctx, tk.phonemes, tk);
                continue;
            }
            size_t left = 0;
            size_t right = w.tks.size();
            bool should_fallback = false;
            while (left < right) {
                bool fixed = false;
                for (size_t k = left; k < right; k++) {
                    fixed |= w.tks[k].alias.has_value() || w.tks[k].phonemes.has_value();
                }
                ps_t ps;
                mtoken merged;
                if (!fixed) {
                    merged = merge_tokens(std::vector<mtoken>(w.tks.begin() + left, w.tks.begin() + right), {});
                    ps = lexicon(merged, ctx);
                }
                if (ps) {
                    w.tks[left].phonemes = ps;
                    for (size_t k = left + 1; k < right; k++) {
                        w.tks[k].phonemes = ustr();
                    }
                    ctx = token_context(ctx, ps, merged);
                    right = left;
                    left = 0;
                } else if (left + 1 < right) {
                    left++;
                } else {
                    right--;
                    mtoken & tk = w.tks[right];
                    if (!tk.phonemes) {
                        if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return has(SUBTOKEN_JUNKS, c); })) {
                            tk.phonemes = ustr();
                        } else if (fallback) {
                            should_fallback = true;
                            break;
                        }
                    }
                    left = 0;
                }
            }
            if (should_fallback) {
                mtoken merged = merge_tokens(w.tks, {});
                w.tks[0].phonemes = run_fallback(merged);
                for (size_t k = 1; k < w.tks.size(); k++) {
                    w.tks[k].phonemes = ustr();
                }
            } else {
                resolve_tokens(w.tks);
            }
        }
        std::vector<mtoken> out;
        for (auto & w : words) {
            mtoken tk = w.is_list ? merge_tokens(w.tks, ustr()) : w.tks[0];
            if (tk.phonemes) {
                // Kokoro v1.0 symbols
                tk.phonemes = replace_all(replace_all(*tk.phonemes, U"ɾ", U"T"), U"ʔ", U"t");
            }
            out.push_back(tk);
        }
        return out;
    }
};

//
// public
//

g2p_en::g2p_en(const char * const * words, const char * const * phonemes, int32_t n, int32_t n_gold, bool british) : p(new impl()) {
    p->british = british;
    std::unordered_map<ustr, lex_entry> golds;
    std::unordered_map<ustr, ustr> silvers;
    for (int32_t i = 0; i < n; i++) {
        const ustr w  = to_u32(words[i]);
        const ustr ph = to_u32(phonemes[i]);
        const ps_t v = ph.empty() ? ps_t() : ps_t(ph);
        if (i < n_gold) {
            const size_t tab = w.find(U'\t');
            if (tab == ustr::npos) {
                golds[w].def = v;
            } else {
                golds[w.substr(0, tab)].tags[to_u8(w.substr(tab + 1))] = v;
            }
        } else if (v) {
            silvers[w] = *v;
        }
    }
    // Lexicon.grow_dictionary(): add the capitalized / lower case forms, existing entries win
    auto grow = [](auto & d) {
        std::vector<std::pair<ustr, typename std::decay_t<decltype(d)>::mapped_type>> e;
        for (const auto & kv : d) {
            const ustr & k = kv.first;
            if (k.size() < 2) {
                continue;
            }
            if (k == lower(k)) {
                if (k != capitalize(k)) {
                    e.push_back({ capitalize(k), kv.second });
                }
            } else if (k == capitalize(lower(k))) {
                e.push_back({ lower(k), kv.second });
            }
        }
        for (auto & kv : e) {
            d.emplace(kv.first, kv.second); // does not overwrite
        }
    };
    grow(golds);
    grow(silvers);
    p->golds   = std::move(golds);
    p->silvers = std::move(silvers);
}

g2p_en::~g2p_en() = default;

std::vector<std::string> g2p_en::phonemize(const std::string & text, const fallback_t & fallback) const {
    const std::vector<mtoken> tokens = p->g2p(to_u32(text), fallback);

    // KPipeline.en_tokenize(): chunks of at most 510 phonemes, split at the last punctuation when possible
    const size_t max_len = 510;
    auto tokens_to_ps = [](const std::vector<mtoken> & tks, size_t n) {
        ustr ps;
        for (size_t i = 0; i < n; i++) {
            ps += (tks[i].phonemes ? *tks[i].phonemes : ustr()) + (tks[i].whitespace.empty() ? U"" : U" ");
        }
        while (!ps.empty() && is_space_c(ps.front())) ps.erase(0, 1);
        while (!ps.empty() && is_space_c(ps.back())) ps.pop_back();
        return ps;
    };
    auto waterfall_last = [&](const std::vector<mtoken> & tks, size_t next_count) {
        for (const ustr & set : { ustr(U"!.?…"), ustr(U":;"), ustr(U",—") }) {
            size_t z = 0;
            bool found = false;
            for (size_t i = tks.size(); i-- > 0; ) {
                const ps_t & ph = tks[i].phonemes;
                if (ph && ph->size() == 1 && has(set, (*ph)[0])) {
                    z = i;
                    found = true;
                    break;
                }
            }
            if (!found) {
                continue;
            }
            z++;
            if (z < tks.size() && tks[z].phonemes && (*tks[z].phonemes == U")" || *tks[z].phonemes == U"”")) {
                z++;
            }
            if (next_count - tokens_to_ps(tks, z).size() <= max_len) {
                return z;
            }
        }
        return tks.size();
    };

    std::vector<std::string> chunks;
    std::vector<mtoken> tks;
    size_t pcount = 0;
    for (mtoken t : tokens) {
        if (!t.phonemes) {
            t.phonemes = ustr();
        }
        ustr next_ps = *t.phonemes + (t.whitespace.empty() ? U"" : U" ");
        ustr trimmed = next_ps;
        while (!trimmed.empty() && is_space_c(trimmed.back())) trimmed.pop_back();
        const size_t next_pcount = pcount + trimmed.size();
        if (next_pcount > max_len) {
            const size_t z = waterfall_last(tks, next_pcount);
            const ustr ps = tokens_to_ps(tks, z);
            if (!ps.empty()) {
                chunks.push_back(to_u8(ps.substr(0, max_len)));
            }
            tks.erase(tks.begin(), tks.begin() + z);
            pcount = tokens_to_ps(tks, tks.size()).size();
            if (tks.empty()) {
                while (!next_ps.empty() && is_space_c(next_ps.front())) next_ps.erase(0, 1);
            }
        }
        tks.push_back(t);
        pcount += next_ps.size();
    }
    if (!tks.empty()) {
        const ustr ps = tokens_to_ps(tks, tks.size());
        if (!ps.empty()) {
            chunks.push_back(to_u8(ps.substr(0, max_len)));
        }
    }
    return chunks;
}

std::string g2p_en::from_espeak(const std::string & ipa, char lang) {
    auto replace_all = [](std::string s, const std::string & from, const std::string & to) {
        for (size_t pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + to.size())) {
            s.replace(pos, from.size(), to);
        }
        return s;
    };
    std::string ps = ipa;
    if (lang == 'a' || lang == 'b') {
        // EspeakFallback.E2M, longest keys first
        static const char * e2m[][2] = {
            { "ʔˌn\u0329", "ʔn" }, { "ʔn\u0329", "ʔn" }, { "a^ɪ", "I" }, { "a^ʊ", "W" }, { "d^ʒ", "ʤ" }, { "e^ɪ", "A" },
            { "t^ʃ", "ʧ" }, { "ɔ^ɪ", "Y" }, { "ə^l", "ᵊl" }, { "ʲo", "jo" }, { "ʲə", "jə" }, { "e", "A" }, { "ʲ", "" },
            { "ɚ", "əɹ" }, { "r", "ɹ" }, { "x", "k" }, { "ç", "k" }, { "ɐ", "ə" }, { "ɬ", "l" }, { "\u0303", "" },
        };
        for (const auto & r : e2m) {
            ps = replace_all(ps, r[0], r[1]);
        }
        // syllabic consonants: X + U+0329 -> schwa X
        const ustr cps = to_u32(ps);
        ustr out;
        for (size_t i = 0; i < cps.size(); i++) {
            if (i + 1 < cps.size() && cps[i + 1] == U'\u0329' && !is_space_c(cps[i])) {
                out += U'ᵊ';
                out += cps[i];
                i++;
            } else if (cps[i] != U'\u0329') {
                out += cps[i];
            }
        }
        ps = to_u8(out);
        if (lang == 'b') {
            ps = replace_all(ps, "e^ə", "ɛː");
            ps = replace_all(ps, "iə", "ɪə");
            ps = replace_all(ps, "ə^ʊ", "Q");
        } else {
            ps = replace_all(ps, "o^ʊ", "O");
            ps = replace_all(ps, "ɜːɹ", "ɜɹ");
            ps = replace_all(ps, "ɜː", "ɜɹ");
            ps = replace_all(ps, "ɪə", "iə");
            ps = replace_all(ps, "ː", "");
        }
        ps = replace_all(ps, "o", "ɔ");
        ps = replace_all(ps, "ɾ", "T");
        ps = replace_all(ps, "ʔ", "t");
        return replace_all(ps, "^", "");
    }
    // language_switch='remove-flags': drop the "(en)" markers around words of another language
    for (size_t b = ps.find('('); b != std::string::npos; b = ps.find('(', b)) {
        const size_t e = ps.find(')', b);
        ps.erase(b, e == std::string::npos ? std::string::npos : e - b + 1);
    }
    // EspeakG2P.e2m, keys in sorted order
    static const char * e2m[][2] = {
        { "a^ɪ", "I" }, { "a^ʊ", "W" }, { "d^z", "ʣ" }, { "d^ʒ", "ʤ" }, { "e^ɪ", "A" }, { "o^ʊ", "O" },
        { "s^s", "S" }, { "t^s", "ʦ" }, { "t^ʃ", "ʧ" }, { "ɔ^ɪ", "Y" }, { "ə^ʊ", "Q" },
    };
    for (const auto & r : e2m) {
        ps = replace_all(ps, r[0], r[1]);
    }
    ps = replace_all(ps, "^", "");
    return replace_all(ps, "-", "");
}
