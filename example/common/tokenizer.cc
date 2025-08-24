#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {
// End-Of-Text
constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    FINISHED：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File does not exist: " << filepath;
    }

    auto file_stream = std::ifstream(filepath, std::ios::binary);
    CHECK(file_stream.is_open()) << "Failed to open file: " << filepath;

    auto header = std::vector<uint8_t>(1024);
    file_stream.read(reinterpret_cast<char*>(header.data()), header.size());

    magic_number_   = BytesToType<uint32_t>(header, 0);
    auto version_num = BytesToType<uint32_t>(header, 4);
    vocab_size_     = BytesToType<uint32_t>(header, 8);

    if (kEotMap.find(magic_number_) == kEotMap.end()) {
        LOG(FATAL) << "Unsupported tokenizer magic number: " << magic_number_;
    }

    eot_token_ = kEotMap.at(magic_number_);

    token_table_.resize(vocab_size_);
    for (auto i = 0; i < vocab_size_; ++i) {
        uint8_t token_len;
        file_stream.read(reinterpret_cast<char*>(&token_len), sizeof(token_len));

        auto buffer = std::vector<char>(token_len);
        file_stream.read(buffer.data(), token_len);

        token_table_[i] = std::string(buffer.data(), token_len);
    }
    for (int i = 0; i < 20; i++) {
        std::cout << "Token[" << i << "] = len=" << (int)token_table_[i].size()
                  << " str=" << token_table_[i] << std::endl;
    }

}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    FINISHED：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    if (token_id >= vocab_size_) {
        return "[ERROR]";
    }
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    // real_seq_len = batch_size * seq_len
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    // host input tensor
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    // sequence_length {prompt,eot,eot...}
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";
    // device
    auto x = std::make_shared<Tensor>(x_tensor.To(device));
    uint64_t kRngState = std::chrono::steady_clock::now().time_since_epoch().count();
    LOG(INFO) << "start generate text:";
    // init default device(CPU0)
    auto cpu_device = Device{};
    // seq {The meaning of life is,generate,generate...,{text_length-1}};
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO:实现单步文本生成逻辑
        HINT:调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // 上一轮结束后x在CPU上,先转移到设备
        x = std::make_shared<infini_train::Tensor>(x->To(device));
        // (bs, seq_len, vocab_size)
        auto logits = model.Forward({x})[0];
        // (bs, seq_len, vocab_size)
        auto probs_device = nn::function::Softmax(logits, 2);
        auto probs_cpu = std::make_shared<Tensor>(probs_device->To(cpu_device));
        auto probs = static_cast<float *>(probs_cpu->DataPtr()) + (t - 1) * logits->Dims()[2];
        auto coin = RandomF32(kRngState);
         auto next = SampleMult(probs, logits->Dims()[2], coin);
        // host
        x = std::make_shared<infini_train::Tensor>(x->To(cpu_device));
        auto add = static_cast<int64_t *>(x->DataPtr());
        add[t] = next;
        std::cout << Decode(next);
    }
    std::cout << std::endl;
}
} // namespace infini_train