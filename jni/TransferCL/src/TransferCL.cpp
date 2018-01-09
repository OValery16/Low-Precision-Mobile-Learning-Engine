#include "TransferCL.h"
#include "../EasyCL/DevicesInfo.h"

#undef STATIC
#define STATIC
#define PUBLIC

//#define TransferCL_VERBOSE 1
//int TransferCL_VERBOSE = 1;

using namespace easycl;


//TransferCL::TransferCL() :
//    EasyCL() {
//}
//TransferCL::TransferCL(int gpu) :
//    EasyCL(gpu) {
//}
PUBLIC TransferCL::TransferCL(cl_platform_id platformId, cl_device_id deviceId) :
    EasyCL(platformId, deviceId) {
}
PUBLIC TransferCL::~TransferCL() {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: ~TransferCL");
#endif


}
PUBLIC void TransferCL::deleteMe() {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: deleteMe");
#endif


    delete this;
}
PUBLIC STATIC TransferCL *TransferCL::createForFirstGpu() {

#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: createForFirstGpu");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedGpu(0, &platformId, &deviceId);
    return new TransferCL(platformId, deviceId);
}
PUBLIC STATIC TransferCL *TransferCL::createForFirstGpuOtherwiseCpu() {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: createForFirstGpuOtherwiseCpu");
#endif


    if(DevicesInfo::getNumGpus() >= 1) {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: getNumGpus() >= 1) {");
#endif


        return createForFirstGpu();
    } else {
        return createForIndexedDevice(0);
    }
}
PUBLIC STATIC TransferCL *TransferCL::createForIndexedDevice(int device) {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: createForIndexedDevice");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedDevice(device, &platformId, &deviceId);
    return new TransferCL(platformId, deviceId);
}
PUBLIC STATIC TransferCL *TransferCL::createForIndexedGpu(int gpu) {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: createForIndexedGpu");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedGpu(gpu, &platformId, &deviceId);
    return new TransferCL(platformId, deviceId);
}
PUBLIC STATIC TransferCL *TransferCL::createForPlatformDeviceIndexes(int platformIndex, int deviceIndex) {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: createForPlatformDeviceIndexes");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedPlatformDevice(platformIndex, deviceIndex, CL_DEVICE_TYPE_ALL, &platformId, &deviceId);
    return new TransferCL(platformId, deviceId);
}
PUBLIC STATIC TransferCL *TransferCL::createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId) {
#if TransferCL_VERBOSE == 1
LOGI( "TransferCL/src/TransferCL.cpp: createForPlatformDeviceIds");
#endif


    return new TransferCL(platformId, deviceId);
}

