// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_backend.hpp"

#include <vector>

#include "intel_npu/config/common.hpp"
#include "zero_device.hpp"

namespace intel_npu {

ZeroEngineBackend::ZeroEngineBackend(const Config& config) : _logger("ZeroEngineBackend", Logger::global().level()) {
    _logger.debug("ZeroEngineBackend - initialize started");

    _instance = std::make_shared<ZeroInitStructsHolder>();

    auto device = std::make_shared<ZeroDevice>(_instance);
    _devices.emplace(std::make_pair(device->getName(), device));
    _logger.debug("ZeroEngineBackend - initialize completed");
}

uint32_t ZeroEngineBackend::getDriverVersion() const {
    return _instance->getDriverVersion();
}

uint32_t ZeroEngineBackend::getGraphExtVersion() const {
    return _instance->getGraphDdiTable().version();
}

bool ZeroEngineBackend::isBatchingSupported() const {
    return _instance->getGraphDdiTable().version() >= ZE_GRAPH_EXT_VERSION_1_6;
}

bool ZeroEngineBackend::isCommandQueueExtSupported() const {
    return _instance->getCommandQueueDdiTable().version() >= ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_0;
}

ZeroEngineBackend::~ZeroEngineBackend() = default;

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice() const {
    if (_devices.empty()) {
        _logger.debug("ZeroEngineBackend - getDevice() returning empty list");
        return {};
    } else {
        _logger.debug("ZeroEngineBackend - getDevice() returning device list");
        return _devices.begin()->second;
    }
}

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice(const std::string& /*name*/) const {
    // TODO Add the search of the device by platform & slice
    return getDevice();
}

const std::vector<std::string> ZeroEngineBackend::getDeviceNames() const {
    _logger.debug("ZeroEngineBackend - getDeviceNames started");
    std::vector<std::string> devicesNames;
    std::for_each(_devices.cbegin(), _devices.cend(), [&devicesNames](const auto& device) {
        devicesNames.push_back(device.first);
    });
    _logger.debug("ZeroEngineBackend - getDeviceNames completed and returning result");
    return devicesNames;
}

void* ZeroEngineBackend::getContext() const {
    return _instance->getContext();
}

void* ZeroEngineBackend::getDriverHandle() const {
    return _instance->getDriver();
}

void* ZeroEngineBackend::getDeviceHandle() const {
    return _instance->getDevice();
}

ze_graph_dditable_ext_curr_t& ZeroEngineBackend::getGraphDdiTable() const {
    return _instance->getGraphDdiTable();
}

void ZeroEngineBackend::updateInfo(const Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    if (_devices.size() > 0) {
        for (auto& dev : _devices) {
            dev.second->updateInfo(config);
        }
    }
}

}  // namespace intel_npu
