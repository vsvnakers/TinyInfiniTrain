#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================

    const auto &input_dims = input->Dims();   // 输入维度
    const auto &other_dims = other->Dims();   // 右乘矩阵维度

    // 保证输入和右乘矩阵至少是二维（最后两维为矩阵形状）
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);

    // 保证矩阵乘法维度可配对：input 的列数 == other 的行数
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);

    // ================================
    // 构造输出张量
    // 形状：保持 input 前面批量维，最后两维变成 M x N
    // ================================
    auto output_dims = input_dims;
    output_dims.back() = other_dims.back();   // 最后一维替换为 N
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    // ================================
    // 取出矩阵的行列数
    // ================================
    const int64_t M = input_dims[input_dims.size() - 2]; // 行数
    const int64_t K = input_dims.back();                 // 内积维
    const int64_t N = other_dims.back();                 // 列数

    // 批次数：总元素数 / 每个矩阵元素数
    const int64_t num_batches        = input->NumElements() / (M * K);
    const int64_t input_matrix_size  = M * K;  // 每个 input 矩阵大小
    const int64_t other_matrix_size  = K * N;  // 每个 other 矩阵大小
    const int64_t output_matrix_size = M * N;  // 每个 output 矩阵大小

    // ================================
    // 获取底层数据指针
    // ================================
    auto *input_data  = static_cast<float *>(input->DataPtr());
    auto *other_data  = static_cast<float *>(other->DataPtr());
    auto *output_data = static_cast<float *>(output->DataPtr());

    // ================================
    // 按批次进行矩阵乘法
    // 使用 Eigen::Map 将裸指针映射为矩阵对象
    // ================================
    for (int64_t i = 0; i < num_batches; ++i) {
        // 映射输入矩阵 A (M x K)
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input_mat(input_data  + i * input_matrix_size,  M, K);

        // 映射右乘矩阵 B (K x N)
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        other_mat(other_data  + i * other_matrix_size,  K, N);

        // 映射输出矩阵 C (M x N)
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        output_mat(output_data + i * output_matrix_size, M, N);

        // 前向公式：C = A @ B
        output_mat = input_mat * other_mat;
    }

    // 返回结果张量
    return output;
    // auto output = std::make_shared<Tensor>();
    // return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================

    // 为两个输入各分配一个梯度张量（与各自形状/设备一致，dtype 为 float32）
    auto grad_input = std::make_shared<Tensor>(input->Dims(),  DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(),  DataType::kFLOAT32, other->GetDevice());

    // --- 形状信息（支持批量维度，最后两维作为矩阵） ---
    const auto &input_dims =  input->Dims();     // ... x M x K
    const auto &other_dims =  other->Dims();     // ... x K x N

    const int64_t M = input_dims[input_dims.size() - 2]; // 行
    const int64_t K = input_dims.back();                 // 公共内积维
    const int64_t N = other_dims.back();                 // 列

    // 批量个数：总元素数 / 每个矩阵元素数（M*K）
    const int64_t num_batches        = input->NumElements() / (M * K);
    const int64_t input_matrix_size  = M * K;  // 每个 input 矩阵的元素数
    const int64_t other_matrix_size  = K * N;  // 每个 other  矩阵的元素数
    const int64_t output_matrix_size = M * N;  // 每个 output 矩阵的元素数

    // --- 原始数据指针（以 float 访问） ---
    auto *input_data       = static_cast<float *>(input->DataPtr());
    auto *other_data       = static_cast<float *>(other->DataPtr());
    auto *grad_output_data = static_cast<float *>(grad_output->DataPtr());
    auto *grad_input_data  = static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_data  = static_cast<float *>(grad_other->DataPtr());

    // --- 按批处理，使用 Eigen::Map 把纯内存映射为矩阵视图（RowMajor） ---
    for (int64_t i = 0; i < num_batches; ++i) {
        // 只读视图：输入矩阵 A（M x K）
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input_mat(      input_data       + i * input_matrix_size,  M, K);

        // 只读视图：权重/右乘矩阵 B（K x N）
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        other_mat(      other_data       + i * other_matrix_size,  K, N);

        // 只读视图：上游梯度 dL/dY（M x N）
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        grad_output_mat(grad_output_data + i * output_matrix_size, M, N);

        // 可写视图：待求的 dL/dA（M x K）
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        grad_input_mat( grad_input_data  + i * input_matrix_size,  M, K);

        // 可写视图：待求的 dL/dB（K x N）
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        grad_other_mat( grad_other_data  + i * other_matrix_size,  K, N);

        // --- 反向公式（矩阵微积分） ---
        // dL/dA = dL/dY * B^T
        grad_input_mat = grad_output_mat * other_mat.transpose();

        // dL/dB = A^T * dL/dY
        grad_other_mat = input_mat.transpose() * grad_output_mat;
    }

    return {grad_input, grad_other};
    
    // auto grad_input = std::make_shared<Tensor>();
    // auto grad_other = std::make_shared<Tensor>();
    // return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
