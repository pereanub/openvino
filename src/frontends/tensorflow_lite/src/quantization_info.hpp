// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "common_op_table.hpp"
#include "decoder_map.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class QuantizationInfo : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("QuantizationInfo");
    QuantizationInfo() = default;
    explicit QuantizationInfo(const std::vector<float>& scale,
                              const std::vector<int64_t>& zero_point,
                              const int64_t& axis)
        : m_scale(scale),
          m_zero_point(zero_point),
          m_axis(axis) {}

    bool is_copyable() const override {
        return false;
    }
    const std::vector<float>& get_scale() const {
        return m_scale;
    }
    void set_scale(const std::vector<float>& scale) {
        m_scale = scale;
    }
    const std::vector<int64_t>& get_zero_point() const {
        return m_zero_point;
    }
    void set_zero_point(const std::vector<int64_t>& zero_point) {
        m_zero_point = zero_point;
    }
    const int64_t& get_axis() const {
        return m_axis;
    }
    void set_axis(const int64_t& axis) {
        m_axis = axis;
    }
    bool is_disabled() const {
        return m_disabled;
    }
    void disable() {
        m_disabled = true;
    }

private:
    std::vector<float> m_scale;
    std::vector<int64_t> m_zero_point;
    int64_t m_axis{};
    bool m_disabled = false;
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
