// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "overload/overload_test_utils_npu.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class RemoteRunTests : public ov::test::behavior::OVPluginTestBase,
                       public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;
    ov::CompiledModel compiled_model;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        ov_model = getDefaultNGraphFunctionForTheDeviceNPU();  // FIXME: E#80555
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }
};

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBuf) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    auto tensor = inference_request.get_input_tensor();
    auto zero_context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    auto remote_tensor =
        zero_context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);
    tensor = {};

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckRemoteTensorInternalBufChangingTensors) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());

    // set input remote tensor
    auto tensor = inference_request.get_input_tensor();
    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto remote_tensor =
        context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random input tensor
    ov::Tensor random_tensor_input{ov::element::f32, tensor.get_shape()};
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(random_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random output tensor
    auto output_tensor = inference_request.get_output_tensor();
    ov::Tensor outputrandom_tensor_input{ov::element::f32, output_tensor.get_shape()};
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(outputrandom_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set output remote tensor
    auto remote_output_tensor = inference_request.get_output_tensor();
    auto output_remote_tensor = context.create_tensor(ov::element::f32, remote_output_tensor.get_shape());
    remote_output_tensor = {};

    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRuns) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    ov::Tensor second_output;

    {
        auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();

        auto remote_tensor =
            context.create_l0_host_tensor(ov::element::f32, tensor.get_shape(), ov::intel_npu::TensorType::INPUT);
        memset(remote_tensor.get(), 1, tensor.get_byte_size());
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
        first_output = inference_request.get_output_tensor(0);
    }

    compiled_model = {};
    inference_request = {};

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();
        float* data = new float[tensor.get_byte_size() / sizeof(float)];
        memset(data, 1, tensor.get_byte_size());
        ov::Tensor input_data_tensor{ov::element::f32, tensor.get_shape(), data};
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());
        second_output = inference_request.get_output_tensor(0);

        delete[] data;
    }

    EXPECT_NE(first_output.data(), second_output.data());
    EXPECT_EQ(memcmp(first_output.data(), second_output.data(), second_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensors1) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    void* first_output = nullptr;
    ov::intel_npu::level_zero::ZeroBufferTensor remote_output_tensor;
    ov::Tensor second_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto input_tensor = inference_request.get_input_tensor();
        auto output_tensor = inference_request.get_output_tensor();
        const auto byte_size = input_tensor.get_byte_size();
        auto input_shape = input_tensor.get_shape();
        auto output_shape = output_tensor.get_shape();
        input_tensor = {};
        output_tensor = {};

        auto remote_input_tensor =
            context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
        remote_output_tensor = context.create_l0_host_tensor(ov::element::f32, output_shape);

        memset(remote_input_tensor.get(), 99, byte_size);
        OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
        OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor));
        OV_ASSERT_NO_THROW(inference_request.infer());

        first_output = remote_output_tensor.get();
    }

    compiled_model = {};
    inference_request = {};

    {
        OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        auto tensor = inference_request.get_input_tensor();
        memset(tensor.data(), 99, tensor.get_byte_size());
        OV_ASSERT_NO_THROW(inference_request.infer());

        second_output = inference_request.get_output_tensor(0);
    }

    EXPECT_NE(first_output, second_output.data());
    EXPECT_EQ(memcmp(first_output, second_output.data(), second_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensors2) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    void* first_output = NULL;
    void* second_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto input_tensor = inference_request.get_input_tensor();
    auto output_tensor = inference_request.get_output_tensor();
    const auto byte_size = input_tensor.get_byte_size();
    auto input_shape = input_tensor.get_shape();
    auto output_shape = output_tensor.get_shape();
    input_tensor = {};
    output_tensor = {};

    auto remote_input_tensor =
        context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
    auto remote_output_tensor1 =
        context.create_tensor(ov::element::f32, output_shape).as<ov::intel_npu::level_zero::ZeroBufferTensor>();

    memset(remote_input_tensor.get(), 99, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor1));
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = remote_output_tensor1.get();

    float* data = new float[byte_size / sizeof(float)];
    memset(data, 99, byte_size);
    ov::Tensor input_data_tensor{ov::element::f32, input_shape, data};
    auto remote_output_tensor2 =
        context.create_tensor(ov::element::f32, output_shape).as<ov::intel_npu::level_zero::ZeroBufferTensor>();

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor2));
    OV_ASSERT_NO_THROW(inference_request.infer());
    second_output = remote_output_tensor2.get();

    EXPECT_NE(first_output, second_output);
    EXPECT_EQ(memcmp(first_output, second_output, remote_output_tensor2.get_byte_size()), 0);

    delete[] data;
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensors3) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;
    void* second_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();
    memset(tensor.data(), 99, tensor.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = inference_request.get_output_tensor(0);

    auto input_tensor = inference_request.get_input_tensor();
    auto output_tensor = inference_request.get_output_tensor();
    const auto byte_size = input_tensor.get_byte_size();
    auto input_shape = input_tensor.get_shape();
    auto output_shape = output_tensor.get_shape();
    input_tensor = {};
    output_tensor = {};

    auto remote_input_tensor =
        context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
    auto remote_output_tensor = context.create_l0_host_tensor(ov::element::f32, output_shape);

    memset(remote_input_tensor.get(), 99, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
    second_output = remote_output_tensor.get();

    EXPECT_NE(first_output.data(), second_output);
    EXPECT_EQ(memcmp(first_output.data(), second_output, first_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensorsHostTensor1) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    ov::Tensor first_output;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();
    memset(tensor.data(), 99, tensor.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.infer());
    first_output = inference_request.get_output_tensor();

    auto l0_host_input_tensor = context.create_host_tensor(ov::element::f32, tensor.get_shape());
    auto l0_host_output_tensor = context.create_host_tensor(ov::element::f32, first_output.get_shape());

    memset(l0_host_input_tensor.data(), 99, tensor.get_byte_size());
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(l0_host_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(l0_host_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(first_output.data(), l0_host_output_tensor.data());
    EXPECT_EQ(memcmp(first_output.data(), l0_host_output_tensor.data(), first_output.get_byte_size()), 0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsInOutRemoteTensorsHostTensor2) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto input_tensor = inference_request.get_input_tensor();
    auto output_tensor = inference_request.get_output_tensor();
    const auto byte_size = input_tensor.get_byte_size();
    auto input_shape = input_tensor.get_shape();
    auto output_shape = output_tensor.get_shape();
    input_tensor = {};
    output_tensor = {};

    auto remote_input_tensor =
        context.create_l0_host_tensor(ov::element::f32, input_shape, ov::intel_npu::TensorType::INPUT);
    auto remote_output_tensor =
        context.create_l0_host_tensor(ov::element::f32, output_shape, ov::intel_npu::TensorType::INPUT);
    memset(remote_input_tensor.get(), 1, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(remote_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto l0_host_input_tensor = context.create_host_tensor(ov::element::f32, input_shape);
    auto l0_host_output_tensor = context.create_host_tensor(ov::element::f32, output_shape);

    memset(l0_host_input_tensor.data(), 99, byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(l0_host_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(l0_host_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(remote_output_tensor.get(), l0_host_output_tensor.data());
    EXPECT_NE(memcmp(remote_output_tensor.get(), l0_host_output_tensor.data(), remote_output_tensor.get_byte_size()),
              0);
}

TEST_P(RemoteRunTests, CheckOutputDataFromTwoRunsOneBigRemoteTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;

    static const std::size_t alignment = 4096;
    std::size_t size = 0;

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto in_tensor = inference_request.get_input_tensor();
    auto out_tensor = inference_request.get_output_tensor();
    const auto input_byte_size = in_tensor.get_byte_size();
    const auto output_byte_size = out_tensor.get_byte_size();
    auto input_shape = in_tensor.get_shape();
    auto output_shape = out_tensor.get_shape();
    in_tensor = {};
    out_tensor = {};

    auto input_byte_size_alligned = (input_byte_size + alignment - 1) & ~(alignment - 1);
    auto output_byte_size_alligned = (output_byte_size + alignment - 1) & ~(alignment - 1);
    size = input_byte_size_alligned + output_byte_size_alligned;

    auto tensor_shape = Shape{1, size};

    auto remote_tensor = context.create_l0_host_tensor(ov::element::f32, tensor_shape);
    auto* input_mem_handle = remote_tensor.get();
    auto* output_mem_handle = static_cast<unsigned char*>(remote_tensor.get()) + input_byte_size_alligned;

    ov::Tensor input_tensor{ov::element::f32, input_shape, input_mem_handle};
    ov::Tensor output_tensor{ov::element::f32, output_shape, output_mem_handle};

    memset(remote_tensor.get(), 1, input_byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    auto l0_host_input_tensor = context.create_host_tensor(ov::element::f32, input_shape);
    auto l0_host_output_tensor = context.create_host_tensor(ov::element::f32, output_shape);

    memset(l0_host_input_tensor.data(), 1, input_byte_size);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(l0_host_input_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(l0_host_output_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(output_mem_handle, l0_host_output_tensor.data());
    EXPECT_EQ(memcmp(output_mem_handle, l0_host_output_tensor.data(), output_byte_size), 0);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
