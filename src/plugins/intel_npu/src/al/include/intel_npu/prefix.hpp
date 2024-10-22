// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace intel_npu {

//
// Prefix for ReadValue and Assign operations in compiler.
//
constexpr std::string_view READVALUE_PREFIX = "vpux_ie_read_value_";
constexpr std::string_view ASSIGN_PREFIX = "vpux_ie_assign_";
constexpr std::string_view SHAPE_TENSOR_PREFIX = "vpux_ie_shape_";


inline bool isStateInputName(const std::string& name) {
    return !name.compare(0, READVALUE_PREFIX.length(), READVALUE_PREFIX);
}
inline bool isStateOutputName(const std::string& name) {
    return !name.compare(0, ASSIGN_PREFIX.length(), ASSIGN_PREFIX);
}
inline bool isShapeTensorName(const std::string& name) {
    return !name.compare(0, SHAPE_TENSOR_PREFIX.length(), SHAPE_TENSOR_PREFIX);
}

inline std::string stateOutputToStateInputName(const std::string& name) {
    return std::string(READVALUE_PREFIX) + name.substr(ASSIGN_PREFIX.length());
}

}  // namespace intel_npu
