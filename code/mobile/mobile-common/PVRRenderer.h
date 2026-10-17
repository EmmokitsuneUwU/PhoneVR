#pragma once

#include <jni.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern float fpsRenderer;

unsigned int PVRInitSystem(int maxWidth,
                           int maxHeight,
                           float offFov,
                           bool reproj,
                           bool debug);   // return render texture id
void PVRRender(int64_t pts);
void PVRTrigger();
void PVRSetImageParams(
    float brightness, float contrast, float gamma, float saturation, float sharpness);
void PVRSetScreenSize(int width, int height);
void PVRPause();
void PVRResume();
void PVRDestroyCardboard();
void PVRCreateCardboard(JavaVM *vm, jobject context, bool reverseLandscape);
void PVRScanViewerQr(JavaVM *vm, jobject context);
bool PVRGetHeadRotation(float *quatWxyz);

#ifdef __cplusplus
}
#endif
