#include "PVRRenderer.h"

#include "Geometry"
#include <algorithm>
#include <ctime>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "PVRSockets.h"
#include "Utils/RenderUtils.h"
#include "cardboard.h"

using namespace std;
using namespace Eigen;

namespace {

    bool reproj = false;
    bool debugMode = false;
    float offFov;

    CardboardHeadTracker *headTracker = nullptr;
    CardboardLensDistortion *lensDistortion = nullptr;
    CardboardDistortionRenderer *distortionRenderer = nullptr;

    bool reverseLandscape = false;
    int screenW = 0, screenH = 0;
    bool distortionDirty = true;
    int lastParamsCount = -1;

    GLuint eyeFbo = 0, eyeTex = 0;
    int eyeW = 0, eyeH = 0;
    float ipd = 0.064f;

    Matrix4f rotInv = Matrix4f::Identity();
    unique_ptr<Renderer> videoRdr[2];

    struct {
        float brightness = 0.f, contrast = 1.f, gamma = 1.f, saturation = 1.f, sharpness = 0.f;
    } imgParams;
    GLint adjLoc[2] = {-1, -1}, sharpLoc[2] = {-1, -1}, texelLoc[2] = {-1, -1};
    float texelW = 0.f, texelH = 0.f;

    const char *const FS_IMG = R"glsl(
        uniform vec4 adj;
        uniform float sharp;
        uniform vec2 texel;
        void main() {
            vec4 t = texture(tex0, coord);
            vec3 c = t.rgb;
            if (sharp > 0.0) {
                vec3 blur = (texture(tex0, coord + vec2(texel.x, 0.0)).rgb +
                             texture(tex0, coord - vec2(texel.x, 0.0)).rgb +
                             texture(tex0, coord + vec2(0.0, texel.y)).rgb +
                             texture(tex0, coord - vec2(0.0, texel.y)).rgb) * 0.25;
                c += (c - blur) * sharp;
            }
            c = (c - 0.5) * adj.y + 0.5 + adj.x;
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
            c = mix(vec3(l), c, adj.w);
            c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / adj.z));
            color = vec4(c, t.a);
        }
    )glsl";

    const float deg2rad = (float) M_PI / 180;
    const float rad2deg = 180.0f / (float) M_PI;
    const float nearP = 0.1;
    const float farP = 10;

    struct Fov {
        float left, right, bottom, top;
    };

    int64_t bootTimeNs() {
        timespec ts;
        clock_gettime(CLOCK_BOOTTIME, &ts);
        return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
    }

    CardboardViewportOrientation viewportOrientation() {
        return reverseLandscape ? kLandscapeRight : kLandscapeLeft;
    }

    Matrix4f Perspective(const Fov &fov) {
        Matrix4f res;
        try {
            res = Matrix4f::Zero();
            float left = -tan(fov.left * deg2rad) * nearP;
            float top = tan(fov.top * deg2rad) * nearP;
            float right = tan(fov.right * deg2rad) * nearP;
            float bottom = -tan(fov.bottom * deg2rad) * nearP;

            res(0, 0) = (2 * nearP) / (right - left);
            res(0, 2) = (right + left) / (right - left);
            res(1, 1) = (2 * nearP) / (top - bottom);
            res(1, 2) = (top + bottom) / (top - bottom);
            res(2, 2) = (nearP + farP) / (nearP - farP);
            res(2, 3) = (2 * nearP * farP) / (nearP - farP);
            res(3, 2) = -1;
        } catch (exception e) {
            PVR_DB_I("PVRRenderer_Perspectivet:: Caught Exception: " + string(e.what()));
        }

        return res;
    }

    struct EyeData {
        int vpLeft, vpBottom, vpWidth, vpHeight;
        Matrix4f quadModel, proj;
    } eyes[2];

    vector<float> leftQuad;

    void createEyeTarget() {
        if (eyeTex) {
            glDeleteTextures(1, &eyeTex);
            eyeTex = 0;
        }
        if (eyeFbo) {
            glDeleteFramebuffers(1, &eyeFbo);
            eyeFbo = 0;
        }

        eyeW = max(screenW / 2, 16);
        eyeH = max(screenH, 16);

        glGenTextures(1, &eyeTex);
        glBindTexture(GL_TEXTURE_2D, eyeTex);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, eyeW * 2, eyeH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenFramebuffers(1, &eyeFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, eyeFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, eyeTex, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // must run on the GL thread
    void rebuildDistortion() {
        try {
            if (screenW <= 0 || screenH <= 0)
                return;

            uint8_t *buffer = nullptr;
            int size = 0;
            CardboardQrCode_getSavedDeviceParams(&buffer, &size);
            bool saved = size > 0;
            if (!saved) {
                CardboardQrCode_destroy(buffer);
                buffer = nullptr;
                CardboardQrCode_getCardboardV1DeviceParams(&buffer, &size);   // do not free
            }

            if (lensDistortion) {
                CardboardLensDistortion_destroy(lensDistortion);
                lensDistortion = nullptr;
            }
            lensDistortion = CardboardLensDistortion_create(buffer, size, screenW, screenH);
            if (saved)
                CardboardQrCode_destroy(buffer);

            if (distortionRenderer) {
                CardboardDistortionRenderer_destroy(distortionRenderer);
                distortionRenderer = nullptr;
            }
            const CardboardOpenGlEsDistortionRendererConfig config{kGlTexture2D};
            distortionRenderer = CardboardOpenGlEs2DistortionRenderer_create(&config);

            createEyeTarget();

            for (int i = 0; i < 2; ++i) {
                CardboardMesh mesh;
                CardboardLensDistortion_getDistortionMesh(lensDistortion, (CardboardEye) i, &mesh);
                CardboardDistortionRenderer_setMesh(distortionRenderer, &mesh, (CardboardEye) i);

                float f[4];   // [left, right, bottom, top] half angles in radians
                CardboardLensDistortion_getFieldOfView(lensDistortion, (CardboardEye) i, f);
                Fov fov = {f[0] * rad2deg, f[1] * rad2deg, f[2] * rad2deg, f[3] * rad2deg};

                EyeData e;
                e.vpLeft = i * eyeW;
                e.vpBottom = 0;
                e.vpWidth = eyeW;
                e.vpHeight = eyeH;
                e.proj = Perspective(fov);

                float quadLeft = -tan((fov.left + offFov) * deg2rad);
                float quadTop =
                    tan((fov.top + offFov * 2 / 3) * deg2rad);   // augment less vertical fov
                float quadRight = tan((fov.right + offFov) * deg2rad);
                float quadBottom = -tan((fov.bottom + offFov * 2 / 3) * deg2rad);
                if (i == kLeft) {
                    leftQuad = {quadLeft, quadTop, quadRight, quadBottom};

                    float m[16];
                    CardboardLensDistortion_getEyeFromHeadMatrix(lensDistortion, kLeft, m);
                    ipd = fabs(m[12]) * 2;   // extracting IPD
                }
                e.quadModel =
                    Affine3f(Translation3f(
                                 (quadRight + quadLeft) / 2.f, (quadTop + quadBottom) / 2.f, -1.f))
                        .matrix() *
                    Affine3f(
                        Scaling((quadRight - quadLeft) / 2.f, (quadTop - quadBottom) / 2.f, 1.f))
                        .matrix();

                eyes[i] = e;
            }

            lastParamsCount = CardboardQrCode_getDeviceParamsChangedCount();
            distortionDirty = false;
        } catch (exception e) {
            PVR_DB_I("PVRRenderer_rebuildDistortion:: Caught Exception: " + string(e.what()));
        }
    }
}   // namespace

unsigned int PVRInitSystem(int maxW, int maxH, float offFov, bool reproj, bool debug) {

    unsigned int videoTex = 0;
    try {
        pvrState = PVR_STATE_INITIALIZATION;

        ::offFov = offFov;
        ::reproj = reproj;
        debugMode = debug;

        rebuildDistortion();

        // block until receive nal headers. todo: optimize timing
        SendAdditionalData({(uint16_t) maxW, (uint16_t) maxH}, leftQuad, ipd);

        videoTex = genTexture(true);
        videoRdr[0].reset(new Renderer({{videoTex, true}}, FS_IMG, 0.0f, 0.5f));
        videoRdr[1].reset(new Renderer({{videoTex, true}}, FS_IMG, 0.5f, 1.0f));

        texelW = 1.0f / (float) maxW;
        texelH = 1.0f / (float) maxH;
        for (int i = 0; i < 2; i++) {
            GLuint p = videoRdr[i]->getProg();
            adjLoc[i] = glGetUniformLocation(p, "adj");
            sharpLoc[i] = glGetUniformLocation(p, "sharp");
            texelLoc[i] = glGetUniformLocation(p, "texel");
        }

        if (headTracker)
            CardboardHeadTracker_resume(headTracker);
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRInitSystem:: Caught Exception: " + string(e.what()));
    }

    return videoTex;
}

void PVRRender(int64_t pts) {
    try {
        PVR_DB("PVRRender:: Rendering " + to_string(pts));
        if (pvrState == PVR_STATE_RUNNING) {
            static Clk::time_point oldtime = Clk::now();

            if (distortionDirty || lastParamsCount != CardboardQrCode_getDeviceParamsChangedCount())
                rebuildDistortion();
            if (!lensDistortion || !distortionRenderer || !headTracker)
                return;

            if (pts > 0) {
                vector<float> v = DequeueQuatAtPts(pts);
                if (v.size() == 4)
                    rotInv.block(0, 0, 3, 3) =
                        Matrix3f(Quaternionf(v[0], v[1], v[2], v[3]));   // todo: simplify
            }

            float pos[3], q[4];
            CardboardHeadTracker_getPose(headTracker,
                                         bootTimeNs() + 20000000,   // 0.020 s
                                         viewportOrientation(),
                                         pos,
                                         q);
            Matrix4f headMat = Matrix4f::Identity();
            headMat.block(0, 0, 3, 3) = Quaternionf(q[3], q[0], q[1], q[2]).toRotationMatrix();
            Matrix4f deltaRot = headMat * rotInv;

            glBindFramebuffer(GL_FRAMEBUFFER, eyeFbo);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_SCISSOR_TEST);
            glViewport(0, 0, eyeW * 2, eyeH);
            if (!debugMode)
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);   // black background
            else
                glClearColor(1.0f, 0.0f, 0.0f, 1.0f);   // red background
            glClear(GL_COLOR_BUFFER_BIT);

            for (int i = 0; i < 2; ++i) {
                auto e = eyes[i];
                glViewport(e.vpLeft, e.vpBottom, e.vpWidth, e.vpHeight);

                Matrix4f mvp;
                if (reproj)
                    mvp = e.proj * deltaRot * e.quadModel;   // order sensitive!!
                else
                    mvp = e.proj * e.quadModel;

                glUseProgram(videoRdr[i]->getProg());
                glUniform4f(adjLoc[i],
                            imgParams.brightness,
                            imgParams.contrast,
                            imgParams.gamma,
                            imgParams.saturation);
                glUniform1f(sharpLoc[i], imgParams.sharpness);
                glUniform2f(texelLoc[i], texelW, texelH);
                videoRdr[i]->render(mvp);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, screenW, screenH);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            CardboardEyeTextureDescription left = {};
            left.texture = eyeTex;
            left.left_u = 0.0f;
            left.right_u = 0.5f;
            left.top_v = 1.0f;
            left.bottom_v = 0.0f;
            CardboardEyeTextureDescription right = left;
            right.left_u = 0.5f;
            right.right_u = 1.0f;

            CardboardDistortionRenderer_renderEyeToDisplay(
                distortionRenderer, 0, 0, 0, screenW, screenH, &left, &right);

            fpsRenderer = (1000000000.0 / (Clk::now() - oldtime).count());
            oldtime = Clk::now();
        }
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRRender:: Caught Exception: " + string(e.what()));
    }
}

void PVRTrigger() {}   // TODO: register press

void PVRSetImageParams(
    float brightness, float contrast, float gamma, float saturation, float sharpness) {
    imgParams.brightness = max(-0.5f, min(0.5f, brightness));
    imgParams.contrast = max(0.5f, min(2.0f, contrast));
    imgParams.gamma = max(0.5f, min(2.0f, gamma));
    imgParams.saturation = max(0.0f, min(2.0f, saturation));
    imgParams.sharpness = max(0.0f, min(2.0f, sharpness));
}

void PVRSetScreenSize(int width, int height) {
    int w = max(width, height), h = min(width, height);
    if (w != screenW || h != screenH) {
        screenW = w;
        screenH = h;
        distortionDirty = true;
    }
}

void PVRPause() {
    try {
        if (pvrState == PVR_STATE_RUNNING && headTracker) {
            CardboardHeadTracker_pause(headTracker);
            pvrState = PVR_STATE_PAUSED;
        }
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRPause:: Caught Exception: " + string(e.what()));
    }
}

void PVRResume() {
    try {
        if (pvrState == PVR_STATE_PAUSED && headTracker) {
            distortionDirty = true;   // rebuilt on the GL thread
            CardboardHeadTracker_resume(headTracker);
            pvrState = PVR_STATE_RUNNING;
        }
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRResume:: Caught Exception: " + string(e.what()));
    }
}

void PVRCreateCardboard(JavaVM *vm, jobject context, bool reverse) {
    try {
        reverseLandscape = reverse;
        Cardboard_initializeAndroid(vm, context);
        if (!headTracker)
            headTracker = CardboardHeadTracker_create();
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRCreateCardboard:: Caught Exception: " + string(e.what()));
    }
}

void PVRDestroyCardboard() {
    try {
        if (headTracker) {
            CardboardHeadTracker_destroy(headTracker);
            headTracker = nullptr;
        }
        if (lensDistortion) {
            CardboardLensDistortion_destroy(lensDistortion);
            lensDistortion = nullptr;
        }
        if (distortionRenderer) {
            CardboardDistortionRenderer_destroy(distortionRenderer);
            distortionRenderer = nullptr;
        }
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRDestroyCardboard:: Caught Exception: " + string(e.what()));
    }
}

void PVRScanViewerQr(JavaVM *vm, jobject context) {
    try {
        Cardboard_initializeAndroid(vm, context);
        CardboardQrCode_scanQrCodeAndSaveDeviceParams();
    } catch (exception e) {
        PVR_DB_I("PVRRenderer_PVRScanViewerQr:: Caught Exception: " + string(e.what()));
    }
}

bool PVRGetHeadRotation(float *quat) {
    if (!headTracker)
        return false;

    float pos[3], q[4];
    CardboardHeadTracker_getPose(headTracker, bootTimeNs(), viewportOrientation(), pos, q);
    quat[0] = q[3];   // w
    quat[1] = q[0];   // x
    quat[2] = q[1];   // y
    quat[3] = q[2];   // z
    return true;
}
