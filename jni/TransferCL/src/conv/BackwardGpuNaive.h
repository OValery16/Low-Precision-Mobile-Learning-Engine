#pragma once

#include "../dependencies.h"

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/templates/TemplatedKernel.h"

#include "../dependencies.h"
#include <iostream>
#include <string>

#include "../../EasyCL/EasyCL.h"
#include "../activate/ActivationFunction.h"
#include "LayerDimensions.h"

#include "../TransferCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuNaive {
public:
	CLKernel *kernel;
    CLKernel *kernelHalf;
    CLKernel *kernel2;

    int globalSize;
    int workgroupsize;

    bool setup;
    bool setupHalf;
    EasyCL *cl;
    LayerDimensions dim;
//    CLKernel *broadcastMultiply;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuNaive();
    VIRTUAL void backward(int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);
    VIRTUAL void backwardHalf(int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);
    BackwardGpuNaive(EasyCL *cl, LayerDimensions dim);
    void setupBuilderBackward(TemplatedKernel *builder);
    void buildKernelBackward( string kernelSource);
    void inferenceBackward(string& kernelSource);
    void  setActivationFunction(TemplatedKernel *builder);

    void setupBuilderBackwardHalf(TemplatedKernel *builder);
    void inferenceBackwardHalf(string& kernelSource);
    void  setActivationFunctionHalf(TemplatedKernel *builder);

    // [[[end]]]
};

