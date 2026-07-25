#!/bin/bash

# MNNCLI Build Script
# This script builds the mnncli executable

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    echo "Usage: $0 [--android-service] [--clean] [--check]"
    echo "  --android-service  Build the arm64 Android plaintext service with LLM/OpenCL/QNN."
    echo "  --clean            Remove only the selected build directories before configuring."
    echo "  --check            Print dependency information after a successful host build."
}

ANDROID_SERVICE=false
CLEAN_BUILD=false
CHECK_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --android-service)
            ANDROID_SERVICE=true
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
        --check)
            CHECK_BUILD=true
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            show_help
            exit 1
            ;;
    esac
done

BUILD_JOBS="${MNN_BUILD_JOBS:-8}"
if ! [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}MNN_BUILD_JOBS must be a positive integer.${NC}"
    exit 1
fi
if [ "$BUILD_JOBS" -gt 8 ]; then
    BUILD_JOBS=8
fi

OS_NAME=$(uname -s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$ANDROID_SERVICE" = true ]; then
    ANDROID_NDK="${ANDROID_NDK:-/root/android_ndk/android-ndk-r29}"
    if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
        echo -e "${RED}Android NDK toolchain not found under: $ANDROID_NDK${NC}"
        exit 1
    fi
    MNN_BUILD_DIR="${MNNCLI_MNN_BUILD_DIR:-$PROJECT_ROOT/build_mnn_static_android_service}"
    MNNCLI_BUILD_DIR="${MNNCLI_BUILD_DIR:-$SCRIPT_DIR/build_mnncli_android_service}"
else
    MNN_BUILD_DIR="${MNNCLI_MNN_BUILD_DIR:-$PROJECT_ROOT/build_mnn_static}"
    MNNCLI_BUILD_DIR="${MNNCLI_BUILD_DIR:-$SCRIPT_DIR/build_mnncli}"
fi

echo -e "${GREEN}Building MNNCLI...${NC}"
echo -e "${YELLOW}Host OS: $OS_NAME${NC}"
echo -e "${YELLOW}Jobs: $BUILD_JOBS${NC}"
echo -e "${YELLOW}MNN build directory: $MNN_BUILD_DIR${NC}"
echo -e "${YELLOW}mnncli build directory: $MNNCLI_BUILD_DIR${NC}"

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "$MNN_BUILD_DIR" "$MNNCLI_BUILD_DIR"
fi

mkdir -p "$MNN_BUILD_DIR" "$MNNCLI_BUILD_DIR"

CMAKE_ARGS=(
    "-DMNN_BUILD_LLM=ON"
    "-DMNN_BUILD_SHARED_LIBS=OFF"
    "-DMNN_SEP_BUILD=OFF"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DMNN_LOW_MEMORY=ON"
    "-DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON"
    "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON"
    "-DLLM_SUPPORT_HTTP_RESOURCE=OFF"
)

MNNCLI_CMAKE_ARGS=(
    "-DMNN_BUILD_DIR=$MNN_BUILD_DIR"
    "-DMNN_SOURCE_DIR=$PROJECT_ROOT"
    "-DCMAKE_BUILD_TYPE=Release"
)

if [ "$ANDROID_SERVICE" = true ]; then
    ANDROID_ARGS=(
        "-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake"
        "-DANDROID_ABI=arm64-v8a"
        "-DANDROID_PLATFORM=android-21"
        "-DANDROID_STL=c++_static"
        "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"
    )
    CMAKE_ARGS+=(
        "${ANDROID_ARGS[@]}"
        "-DMNN_USE_LOGCAT=OFF"
        "-DMNN_USE_SSE=OFF"
        "-DMNN_BUILD_TEST=OFF"
        "-DMNN_BUILD_BENCHMARK=OFF"
        "-DMNN_BUILD_TRAIN=OFF"
        "-DMNN_BUILD_CONVERTER=OFF"
        "-DMNN_BUILD_LLM_OMNI=OFF"
        "-DLLM_SUPPORT_VISION=OFF"
        "-DLLM_SUPPORT_AUDIO=OFF"
        "-DMNN_BUILD_OPENCV=OFF"
        "-DMNN_IMGCODECS=OFF"
        "-DMNN_BUILD_AUDIO=OFF"
        "-DMNN_BUILD_DIFFUSION=OFF"
        "-DMNN_OPENCL=ON"
        "-DMNN_QNN=ON"
        "-DMNN_WITH_PLUGIN=ON"
        "-DMNN_QNN_ONLINE_FINALIZE=ON"
        "-DMNN_BUILD_FOR_ANDROID_COMMAND=ON"
    )
    MNNCLI_CMAKE_ARGS+=(
        "${ANDROID_ARGS[@]}"
        "-DMNNCLI_ENABLE_TLS=OFF"
        "-DMNNCLI_SERVICE_ONLY=ON"
    )
elif [ "$OS_NAME" = "Darwin" ]; then
    CMAKE_ARGS+=(
        "-DLLM_SUPPORT_VISION=ON"
        "-DMNN_BUILD_OPENCV=ON"
        "-DMNN_IMGCODECS=ON"
        "-DLLM_SUPPORT_AUDIO=ON"
        "-DMNN_BUILD_AUDIO=ON"
        "-DMNN_BUILD_DIFFUSION=ON"
        "-DMNN_USE_OPENCV=ON"
        "-DMNN_METAL=ON"
    )
    SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
    MNNCLI_CMAKE_ARGS+=("-DCMAKE_OSX_SYSROOT=$SDK_PATH")
elif [ "$OS_NAME" = "Linux" ]; then
    CMAKE_ARGS+=(
        "-DLLM_SUPPORT_VISION=ON"
        "-DMNN_BUILD_OPENCV=ON"
        "-DMNN_IMGCODECS=ON"
        "-DLLM_SUPPORT_AUDIO=ON"
        "-DMNN_BUILD_AUDIO=ON"
        "-DMNN_BUILD_DIFFUSION=ON"
        "-DMNN_USE_OPENCV=ON"
        "-DMNN_OPENCL=ON"
    )
else
    echo -e "${YELLOW}Warning: unknown host OS $OS_NAME; using common options.${NC}"
fi

echo -e "${BLUE}Stage 1/2: configuring and building static MNN${NC}"
cmake -B "$MNN_BUILD_DIR" -S "$PROJECT_ROOT" "${CMAKE_ARGS[@]}"
cmake --build "$MNN_BUILD_DIR" --target MNN -j"$BUILD_JOBS"

if [ ! -f "$MNN_BUILD_DIR/libMNN.a" ]; then
    echo -e "${RED}Failed to build static MNN: $MNN_BUILD_DIR/libMNN.a${NC}"
    exit 1
fi

echo -e "${BLUE}Stage 2/2: configuring and building mnncli${NC}"
cmake -B "$MNNCLI_BUILD_DIR" -S "$SCRIPT_DIR" "${MNNCLI_CMAKE_ARGS[@]}"
cmake --build "$MNNCLI_BUILD_DIR" --target mnncli -j"$BUILD_JOBS"

if [ ! -f "$MNNCLI_BUILD_DIR/mnncli" ]; then
    echo -e "${RED}Build failed: $MNNCLI_BUILD_DIR/mnncli was not created.${NC}"
    exit 1
fi

if [ "$ANDROID_SERVICE" = true ]; then
    STRIP_TOOL="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
    if [ -x "$STRIP_TOOL" ]; then
        "$STRIP_TOOL" "$MNNCLI_BUILD_DIR/mnncli"
    fi
fi

ls -lh "$MNN_BUILD_DIR/libMNN.a" "$MNNCLI_BUILD_DIR/mnncli"
if [ "$ANDROID_SERVICE" = false ]; then
    "$MNNCLI_BUILD_DIR/mnncli" --help >/dev/null
    if [ "$CHECK_BUILD" = true ]; then
        if [ "$OS_NAME" = "Darwin" ]; then
            otool -L "$MNNCLI_BUILD_DIR/mnncli"
        elif [ "$OS_NAME" = "Linux" ]; then
            ldd "$MNNCLI_BUILD_DIR/mnncli"
        fi
    fi
else
    file "$MNNCLI_BUILD_DIR/mnncli"
fi

echo -e "${GREEN}Build completed successfully: $MNNCLI_BUILD_DIR/mnncli${NC}"
