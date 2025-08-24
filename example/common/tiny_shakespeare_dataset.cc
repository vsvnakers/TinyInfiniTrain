#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       FINISHED：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    // 检查文件是否存在
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File does not exist: " << path;
    }

    // 以二进制方式打开文件输入流
    auto file_stream = std::ifstream(path, std::ios::binary);
    CHECK(file_stream.is_open()) << "Failed to open dataset file";

    // 读取文件头（固定 1024 字节）
    auto header = std::vector<uint8_t>(1024);
    file_stream.read(reinterpret_cast<char *>(header.data()), header.size());

    // 解析文件头中的关键信息：magic、version、token 总数
    auto file_magic   = BytesToType<int32_t>(header, 0);  // 前 4B: 魔数
    auto file_version = BytesToType<int32_t>(header, 4);  // 4-7B: 版本号
    auto token_count  = BytesToType<int32_t>(header, 8);  // 8-11B: token 总数

    // 校验 magic 是否在类型映射表中
    CHECK(kTypeMap.count(file_magic)) << "Unknown file magic code";
    // 获取 token 类型和对应字节数
    auto token_kind      = kTypeMap.at(file_magic);
    auto token_byte_size = kTypeToSize.at(token_kind);

    // 计算样本数量和样本维度 (sample_count × sequence_length)
    auto sample_count = token_count / static_cast<int64_t>(sequence_length);
    auto sample_dims  = std::vector<int64_t>{sample_count, static_cast<int64_t>(sequence_length)};

    // 构建返回的 TinyShakespeareFile 对象
    auto dataset_file   = TinyShakespeareFile{};
    dataset_file.type   = token_kind;                          // 数据类型
    dataset_file.dims   = sample_dims;                         // 数据维度
    dataset_file.tensor = infini_train::Tensor(sample_dims,    // 分配 tensor
                                            DataType::kINT64);
    auto *tensor_ptr    = static_cast<int64_t *>(dataset_file.tensor.DataPtr());

    // 总的 token 数（样本数 × 每个序列长度）
    auto total_token_elements = sample_count * sequence_length;

    // 根据 token 类型读取数据并写入 tensor
    if (token_kind == TinyShakespeareType::kUINT16) {
        // 限制 UINT16 的最大序列长度
        CHECK(sequence_length <= 1024) << "Sequence length exceeds UINT16 max";
        // 读取所有 token (uint16_t)
        auto raw_tokens = std::vector<uint16_t>(total_token_elements);
        file_stream.read(reinterpret_cast<char *>(raw_tokens.data()),
                        raw_tokens.size() * sizeof(uint16_t));
        // 转换为 int64 存入 tensor
        for (auto i = 0u; i < raw_tokens.size(); ++i) {
            tensor_ptr[i] = static_cast<int64_t>(raw_tokens[i]);
        }
    } else if (token_kind == TinyShakespeareType::kUINT32) {
        // 限制 UINT32 的最大序列长度
        CHECK(sequence_length <= 8192) << "Sequence length exceeds UINT32 max";
        // 读取所有 token (int32_t)
        auto raw_tokens = std::vector<int32_t>(total_token_elements);
        file_stream.read(reinterpret_cast<char *>(raw_tokens.data()),
                        raw_tokens.size() * sizeof(int32_t));
        // 转换为 int64 存入 tensor
        for (auto i = 0u; i < raw_tokens.size(); ++i) {
            tensor_ptr[i] = static_cast<int64_t>(raw_tokens[i]);
        }
    } else {
        // 不支持的 token 类型
        LOG(FATAL) << "Unsupported token type in dataset";
    }

    // 返回数据集文件对象
    return dataset_file;

}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)), num_samples_(text_file_.dims[0] - 1) {
    // =================================== 作业 ===================================
    // FINISHED：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    CHECK_EQ(text_file_.dims[1], sequence_length_);
    CHECK_EQ(static_cast<int>(text_file_.tensor.Dtype()), static_cast<int>(DataType::kINT64));
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }