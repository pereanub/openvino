// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cid_graph.hpp"

#include "intel_npu/config/runtime.hpp"

namespace intel_npu {

CidGraph::CidGraph(const std::shared_ptr<IZeroLink>& zeroLink,
                   ze_graph_handle_t graphHandle,
                   NetworkMetadata metadata,
                   const Config& config)
    : IGraph(graphHandle, std::move(metadata)),
      _zeroLink(zeroLink),
      _config(config),
      _logger("CidGraph", _config.get<LOG_LEVEL>()) {
    if (_config.get<CREATE_EXECUTOR>()) {
        initialize();
    } else {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
    }
}

CompiledNetwork CidGraph::export_blob() const {
    return _zeroLink->getCompiledNetwork(_handle);
}

std::vector<ov::ProfilingInfo> CidGraph::process_profiling_output(const std::vector<uint8_t>& profData) const {
    OPENVINO_THROW("Profiling post-processing is not supported.");
}

void CidGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_zeroLink == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeroLink->setArgumentValue(_handle, argi, argv);
}

void CidGraph::initialize() {
    if (_zeroLink) {
        _logger.debug("Graph initialize start");

        std::tie(_input_descriptors, _output_descriptors) = _zeroLink->getIODesc(_handle);
        _command_queue = _zeroLink->crateCommandQueue(_config);

        if (_config.has<WORKLOAD_TYPE>()) {
            setWorkloadType(_config.get<WORKLOAD_TYPE>());
        }

        _zeroLink->graphInitialie(_handle, _config);

        _logger.debug("Graph initialize finish");
    }
}

CidGraph::~CidGraph() {
    if (_handle != nullptr) {
        auto result = _zeroLink->release(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }
}

}  // namespace intel_npu
