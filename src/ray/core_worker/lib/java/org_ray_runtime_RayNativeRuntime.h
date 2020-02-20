/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_ray_runtime_RayNativeRuntime */

#ifndef _Included_org_ray_runtime_RayNativeRuntime
#define _Included_org_ray_runtime_RayNativeRuntime
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeInitCoreWorker
 * Signature:
 * (ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;I[BLorg/ray/runtime/gcs/GcsClientOptions;)J
 */
JNIEXPORT jlong JNICALL Java_org_ray_runtime_RayNativeRuntime_nativeInitCoreWorker(
    JNIEnv *, jclass, jint, jstring, jstring, jstring, jint, jbyteArray, jobject);

/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeRunTaskExecutor
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_ray_runtime_RayNativeRuntime_nativeRunTaskExecutor(JNIEnv *, jclass, jlong);

/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeDestroyCoreWorker
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_ray_runtime_RayNativeRuntime_nativeDestroyCoreWorker(JNIEnv *, jclass, jlong);

/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeSetup
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_ray_runtime_RayNativeRuntime_nativeSetup(JNIEnv *, jclass,
                                                                         jstring);

/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeShutdownHook
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_ray_runtime_RayNativeRuntime_nativeShutdownHook(JNIEnv *,
                                                                                jclass);

/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeSetResource
 * Signature: (JLjava/lang/String;D[B)V
 */
JNIEXPORT void JNICALL Java_org_ray_runtime_RayNativeRuntime_nativeSetResource(
    JNIEnv *, jclass, jlong, jstring, jdouble, jbyteArray);

/*
 * Class:     org_ray_runtime_RayNativeRuntime
 * Method:    nativeKillActor
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_org_ray_runtime_RayNativeRuntime_nativeKillActor(JNIEnv *,
                                                                             jclass,
                                                                             jlong,
                                                                             jbyteArray);

#ifdef __cplusplus
}
#endif
#endif
