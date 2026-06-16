adb shell mkdir -p /data/local/tmp/mnn-dynamic-draft
adb push libMNN.so /data/local/tmp/mnn-dynamic-draft
adb push libMNN_CL.so /data/local/tmp/mnn-dynamic-draft
adb push libMNN_Express.so /data/local/tmp/mnn-dynamic-draft
adb push libllm.so /data/local/tmp/mnn-dynamic-draft
adb push llm_demo /data/local/tmp/mnn-dynamic-draft
adb push spec_eval /data/local/tmp/mnn-dynamic-draft
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so /data/local/tmp/mnn-dynamic-draft
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so /data/local/tmp/mnn-dynamic-draft
adb push ${QNN_SDK_ROOT}/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so /data/local/tmp/mnn-dynamic-draft
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so /data/local/tmp/mnn-dynamic-draft

