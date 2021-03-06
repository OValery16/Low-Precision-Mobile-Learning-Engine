// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
#include "../../EasyCL/templates/TemplatedKernel.h"
#include "../dependencies.h"
#include <algorithm>
#include <iostream>
#include <string>

#include "../../EasyCL/EasyCL.h"
#include "../activate/ActivationFunction.h"
#include "LayerDimensions.h"
#include "../TransferCLDllExport.h"
#include "table_str.h"

#define TEST_FORWARD 0 // if equal to 1, activate comparison implementation

class AddBias;

class Forward1 {
public:
    EasyCL *cl;
    LayerDimensions dim;

	#if TEST_FORWARD==1
		CLKernel *kernel;
		CLKernel *kernelH;
		AddBias *addBias;
	#endif

    CLKernel *test;
    CLKernel *test2;
    CLKernel *test3;
    CLKernel *test4;

    ///////////////////
    //quatify parameter
	float multR=0.0f;
	float minR=0.0f;
	float multW=0.0f;
	float minW=0.0f;
	float multI=0.0f;
	float minI=0.0f;

    //////////////////

    bool setup;
    bool setupHalf;
    bool setupEightBit;
    int batchSize;
    int iteration;
//    CLWrapper *dataWrapper;
//    CLWrapper *weightsWrapper;
//    CLWrapper *outputWrapper;
    int globalSize;
    int workgroupsize;
    bool normalization;
	#if 1//TEST_FORWARD==1
		const bool timeBenchmark=true;
		clock_t startTimer1, stopTimer1;
	#endif

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward1();
    void setAutoVectorization(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString);
    void setHintCompiler(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder);
    void setPoolingLayer(string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder);
    void  setActivationFunction(TemplatedKernel *builder);
    void testCondition(bool &ok1);
    void writeKernelcode(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString,bool fullvectorization, int batchSize,bool ok1);
    void setInternalLoop(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad);
    void setNonPoolingLayerVariable(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization);

    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper);

    VIRTUAL void forwardFloat(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
        CLWrapper *outputWrapper);
    virtual void forwardHalf(int batchSize,CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
       CLWrapper *outputWrapper, CLWrapper *poolingSelectorWrapper, CLWrapper *gradInputWrapper);

    void buildKernelConvolve(int batchSize);
    void setupBuilderConvolve(TemplatedKernel *builder,int batchSize);
    STATIC std::string getKernelTemplateConvolve();
    Forward1(bool needToNormalize,int batchSize,EasyCL *cl, LayerDimensions dim);

    //////////////
    void writeKernelcodeHalf(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString,bool fullvectorization, int batchSize,bool ok1);
    void setInternalLoopHalf(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad);
    void setHintCompilerHalf(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder);
    void setAutoVectorizationHalf(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString);
    void setPoolingLayerHalf(string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder);
    void setActivationFunctionHalf(TemplatedKernel *builder);
    void setNonPoolingLayerVariableHalf(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization);
    void testConditionHalf(bool &ok1);
    void setupBuilderConvolveHalf(TemplatedKernel *builder,int batchSize);
    STATIC std::string getKernelTemplateConvolveHalf();
    //////////////
    void writeKernelcodeEightBit(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString,bool fullvectorization, int batchSize,bool ok1);
    void setInternalLoopEightBit(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad);
    void setHintCompilerEightBit(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder);
    void setAutoVectorizationEightBit(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString);
    void setPoolingLayerEightBit(string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder);
    void setActivationFunctionEightBit(TemplatedKernel *builder);
    void setNonPoolingLayerVariableEightBit(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization);
    void testConditionEightBit(bool &ok1);
    void setupBuilderConvolveEightBit(TemplatedKernel *builder,int batchSize);
    STATIC std::string getKernelTemplateConvolveEightBit(LayerDimensions dim);
    void ExtractMinMultFromFile(float &minW, float &multW,float &minI, float &multI,float &minR, float &multR);



    void getLastFCKernel(CLWrapper * dataWrapper);

#if TEST_FORWARD==1
    float *convolv(int batchSize, float *inputData, float *weights);
#endif

    void testFloat(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
        CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput);
    void testHalf(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
           CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput);
    VIRTUAL void forwardChooseMethod(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
        CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper *weightsWrapper2, CLWrapper *biasWrapper2, CLWrapper * tempOutput, CLWrapper *gradInputWrapperHalf,CLWrapper *dataWrapper8Bit, CLWrapper *weightsWrapper8Bit, CLWrapper *biasWrapper8Bit,
        CLWrapper *outputWrapper8Bit);

    VIRTUAL int getWorkgroupSize(int globalSize,int maxWorkgroupSizeSize);
    VIRTUAL void forward8bitTestVersion(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput);
    VIRTUAL void forward8bit(int batchSize, CLWrapper *dataWrapper8Bit, CLWrapper *weightsWrapper8Bit, CLWrapper *biasWrapper,CLWrapper *outputWrapper8Bit, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper);
    VIRTUAL void ExtractMinMultFromData(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput);

    //STATIC std::string getTHalf(int i);

    // [[[end]]]
};

