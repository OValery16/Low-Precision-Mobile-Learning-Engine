// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
//#include "BackpropWeights.h"
#include "../../EasyCL/templates/TemplatedKernel.h"

#include "../../EasyCL/EasyCL.h"
#include "../activate/ActivationFunction.h"
#include "LayerDimensions.h"
#include "../TransferCLDllExport.h"

#define MEASURE_BACKWARD_PROP 0
#define TEST_KERNEL 0

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsNaive {
public:
    CLKernel *kernelHalf;
    CLKernel *kernel2;

    CLKernel *kernelTestHalf;
    CLKernel *kernelTest1;
    int globalSize = 0;

    EasyCL *cl;
    LayerDimensions dim;
    int method=1;

    int workgroupsize = 0;//cl->getMaxWorkgroupSize();
    bool setup;
    bool setupHalf;
    const float learningMultiplier;
    ///////////////////
    //quatify parameter
	float multR=0.0f;
	float minR=0.0f;
	float multW=0.0f;
	float minW=0.0f;
	float multI=0.0f;
	float minI=0.0f;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeightsNaive();
    float learningRateToMultiplier(int batchSize);
    VIRTUAL void calcGradWeightsHalf(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper, CLWrapper *weightWrapper, CLWrapper *previousStepVectorWrapper, CLWrapper *biasWrapper, CLWrapper *previousStepBiasVectorWrapper);
    VIRTUAL void calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper, CLWrapper *weightWrapper, CLWrapper *previousStepVectorWrapper, CLWrapper *biasWrapper, CLWrapper *previousStepBiasVectorWrapper);
    BackpropWeightsNaive(EasyCL *cl, LayerDimensions dim);
    void buildKernelBackward( string kernelSource);
    void setupBuilderBackward(TemplatedKernel *builder);
    void setHintCompiler(int batchSize, TemplatedKernel *builder);
    void setupUpdateWeight(string &updateWeight,string &updateBiasWeights, string &defineVariableUpdates, string & updateBiasWeight);
    void setupUpdateWeightVectorized(string &updateWeight, string & updateBiasWeight,string remainerString2,string remainerString3,bool selector);

    //////////////////
    void buildKernelBackwardHalf( string kernelSource);
	void setupBuilderBackwardHalf(TemplatedKernel *builder);
	void setHintCompilerHalf(int batchSize, TemplatedKernel *builder);
	void setupUpdateWeightHalf(string &updateWeight,string &updateBiasWeights, string &defineVariableUpdates, string & updateBiasWeight);
	void setupUpdateWeightVectorizedHalf(string &updateWeight, string & updateBiasWeight,string remainerString2,string remainerString3,bool selector);
	VIRTUAL void ExtractMinMultFromFile(float &minW, float &multW,float &minI, float &multI,float &minR, float &multR);
	// [[[end]]]
};

