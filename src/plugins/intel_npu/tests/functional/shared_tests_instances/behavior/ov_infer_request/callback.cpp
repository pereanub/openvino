// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/callback.hpp"

#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_infer_request/callback.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                               ov::device::properties(ov::test::utils::DEVICE_NPU, ov::AnyMap{})}};

const std::vector<ov::AnyMap> autoConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                              ov::device::properties(ov::test::utils::DEVICE_NPU, ov::AnyMap{})}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Multi_BehaviorTests,
                         OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Auto_BehaviorTests,
                         OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVInferRequestCallbackTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Multi_BehaviorTests,
                         OVInferRequestCallbackTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Auto_BehaviorTests,
                         OVInferRequestCallbackTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

}  // namespace
