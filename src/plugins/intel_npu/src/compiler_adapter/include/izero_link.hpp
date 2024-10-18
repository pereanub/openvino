// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "driver_compiler_utils.hpp"
#include "igraph.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "npu.hpp"

namespace intel_npu {

class IZeroLink {
public:
    virtual std::unordered_set<std::string> queryResultFromSupportedLayers(SerializedIR serializedIR,
                                                                           const std::string& buildFlags) const = 0;

    virtual ze_graph_handle_t getGraphHandle(SerializedIR serializedIR,
                                             const std::string& buildFlags,
                                             const uint32_t& flags) const = 0;

    virtual ze_graph_handle_t getGraphHandle(const std::vector<uint8_t>& network) const = 0;

    virtual NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const = 0;

    virtual _ze_result_t release(ze_graph_handle_t graphHandle) = 0;

    virtual CompiledNetwork getCompiledNetwork(ze_graph_handle_t graphHandle) = 0;

    virtual void setArgumentValue(ze_graph_handle_t graphHandle, uint32_t argi_, const void* argv) const = 0;

    virtual void graphInitialie(ze_graph_handle_t graphHandle, const Config& config) const = 0;

    virtual std::tuple<std::vector<ArgumentDescriptor>, std::vector<ArgumentDescriptor>> getIODesc(
        ze_graph_handle_t graphHandle) const = 0;

    virtual std::shared_ptr<CommandQueue> crateCommandQueue(const Config& config) const = 0;

    virtual ~IZeroLink() = default;
};

}  // namespace intel_npu
