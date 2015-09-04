LOCAL_PATH := $(call my-dir)

include $(NVIDIA_DEFAULTS)

LOCAL_MODULE_TAGS := optional
LOCAL_MODULE := cudnn
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := FileUtils_Android.cpp  Main.cpp  Network.cpp

ifeq ($(TARGET_ARCH),arm64)
  LOCAL_C_INCLUDES   += $(LOCAL_PATH)/../android-cudnn-v3/cuda/include
endif

LOCAL_STATIC_LIBRARIES += \
		libcudnn_static \
		libcudart_static \
		libcublas_static \
		libculibos \
		libgnustl_static

LOCAL_LDLIBS := -llog -landroid -ldl

LOCAL_SDK_VERSION := 9
LOCAL_NDK_STL_VARIANT := gnustl_static
LOCAL_CPPFLAGS := -fexceptions

include $(NVIDIA_EXECUTABLE)
include $(call all-makefiles-under, $(LOCAL_PATH))
