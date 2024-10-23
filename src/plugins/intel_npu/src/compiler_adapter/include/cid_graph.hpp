// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "izero_link.hpp"

namespace intel_npu {

class CidGraph final : public IGraph {
public:
    CidGraph(const std::shared_ptr<IZeroLink>& zeroLink,
             ze_graph_handle_t graphHandle,
             NetworkMetadata metadata,
             const Config& config);

    CompiledNetwork export_blob() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize() override;

    ~CidGraph() override;

private:
    std::shared_ptr<IZeroLink> _zeroLink;

    const Config _config;
    Logger _logger;
};

}  // namespace intel_npu
