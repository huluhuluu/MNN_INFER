//
//  QNNPerf.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNPerf.hpp"

namespace MNN {
namespace QNN {

QNNPerf::QNNPerf(const QNN_INTERFACE_VER_TYPE * qnnInterface) {
    if (qnnInterface == nullptr || qnnInterface->deviceGetInfrastructure == nullptr) {
        MNN_ERROR("MNN_QNN: QNN performance infrastructure is unavailable.\n");
        return;
    }
    mQnnInterface = qnnInterface;

    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    const int infrastructureError = mQnnInterface->deviceGetInfrastructure(&deviceInfra) & 0xFFFF;
    if (infrastructureError != QNN_SUCCESS || deviceInfra == nullptr) {
        MNN_ERROR("MNN_QNN: Failed to get QNN performance infrastructure, error code %d.\n",
                  infrastructureError);
        return;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra  = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    mPerfInfra = htpInfra->perfInfra;
    if (mPerfInfra.createPowerConfigId == nullptr ||
        mPerfInfra.destroyPowerConfigId == nullptr ||
        mPerfInfra.setPowerConfig == nullptr) {
        MNN_ERROR("MNN_QNN: QNN performance control functions are unavailable.\n");
        return;
    }

    uint32_t deviceId = 0;
    uint32_t coreId   = 0;
    const int powerConfigError = mPerfInfra.createPowerConfigId(deviceId, coreId, &mPowerConfigId) & 0xFFFF;
    if (powerConfigError != QNN_SUCCESS) {
        MNN_ERROR("MNN_QNN: Failed to create QNN power config, error code %d.\n", powerConfigError);
        return;
    }
    mValid = true;

    mPowerConfigBurst = {
        .option       = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId               = mPowerConfigId,                                           //use the power config id created
            .setDcvsEnable           = 1,
            .dcvsEnable              = 0,                                                        //1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
            .powerMode               = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            .setSleepLatency         = 1,                                                        //True to consider Latency parameter otherwise False
            .sleepLatency            = 40,                                                       // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
            .setSleepDisable         = 1,                                                        //True to consider sleep disable/enable parameter otherwise False
            .sleepDisable            = 1,                                                        //True to disable sleep, False to re-enable sleep
            .setBusParams            = 1,                                                        //True to consider Bus parameter otherwise False
            .busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .setCoreParams           = 1,                                                        //True to consider Core parameter otherwise False
            .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
        },
    };

    mPowerConfigBalanced = {
        .option       = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId               = mPowerConfigId,                                         //use the power config id created
            .setDcvsEnable           = 1,
            .dcvsEnable              = 1,                                                      //1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
            .powerMode               = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
            .setSleepLatency         = 1,                                                      //True to consider Latency parameter otherwise False
            .sleepLatency            = 1000,                                                   // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
            .setSleepDisable         = 1,                                                      //True to consider sleep disable/enable parameter otherwise False
            .sleepDisable            = 0,                                                      //True to disable sleep, False to re-enable sleep
            .setBusParams            = 1,                                                      //True to consider Bus parameter otherwise False
            .busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_TURBO,
            .setCoreParams           = 1,                                                      //True to consider Core parameter otherwise False
            .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO,
        },
    };

}


// destory power config
QNNPerf::~QNNPerf() {
    if (mValid) {
        CALL_QNN(mPerfInfra.destroyPowerConfigId(mPowerConfigId));
    }
}


void QNNPerf::setRpcLatencyAndPolling() {
    if (!mValid) {
        return;
    }
    // set RPC Control Latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;            // refer QnnHtpPerfInfrastructure.h
    ::memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;         // use rpc control latency recommended 100 us, refer hexagon sdk
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {&rpcControlLatency, NULL};

    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs1));  // set RPC latency config on power config ID created

    // set RPC Polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;   // refer QnnHtpPerfInfrastructure.h
    ::memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999;     // use rpc polling time recommended 0-10000 us
    const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, NULL};

    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs2)); // set RPC polling config on power config ID created
}

void QNNPerf::setPowerConfigBurst() {
    if (!mValid) {
        return;
    }
    #ifdef QNN_VERBOSE
    MNN_PRINT("MNN QNN set burst mode\n");
    #endif
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&mPowerConfigBurst, NULL};
    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs));
}

void QNNPerf::setPowerConfigBalanced() {
    if (!mValid) {
        return;
    }
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&mPowerConfigBalanced, NULL};
    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs));
}

} // end namespace QNN
} // end namespace MNN
