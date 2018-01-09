// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "InputLayerMaker.h"

#include "InputLayer.h"

#include "../dependencies.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

InputLayer::InputLayer(InputLayerMaker *maker) :
       Layer(0, maker),
       cl(maker->cl),
    batchSize(0),
    allocatedSize(0),
    outputPlanes(maker->_numPlanes),
    outputSize(maker->_imageSize),
    input(0),
    output(0),
    outputWrapper(0),
    setup(false){
}
VIRTUAL InputLayer::~InputLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: ~InputLayer");
#endif
#if EIGHT_BIT_ACCURACY==1
	delete outputWrapper8Bit;
#else
#if HALF_ACCURACY==0
	delete outputWrapper;
#else
	delete outputWrapperHalf;
#endif
#endif
}
VIRTUAL std::string InputLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: string InputLayer::getClassName");
#endif


    return "InputLayer";
}
VIRTUAL float *InputLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: getOutput");
#endif


    return output;
}
VIRTUAL bool InputLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: needsBackProp");
#endif


    return false;
}
VIRTUAL int InputLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL void InputLayer::printOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: printOutput");
#endif


    if(output == 0) {
         return;
    }
    for(int n = 0; n < std::min(5,batchSize); n++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: min(5,batchSize); n++) {");
#endif


        std::cout << "InputLayer n " << n << ":" << std::endl;
        for(int plane = 0; plane < std::min(5, outputPlanes); plane++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: min(5, outputPlanes); plane++) {");
#endif


            if(outputPlanes > 1) std::cout << "    plane " << plane << ":" << std::endl;
            for(int i = 0; i < std::min(5, outputSize); i++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: min(5, outputSize); i++) {");
#endif


                std::cout << "      ";
                for(int j = 0; j < std::min(5, outputSize); j++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: min(5, outputSize); j++) {");
#endif


                    std::cout << getOutput(n, plane, i, j) << " ";
//output[
//                            n * numPlanes * imageSize*imageSize +
//                            plane*imageSize*imageSize +
//                            i * imageSize +
//                            j ] << " ";
                }
                if(outputSize > 5) std::cout << " ... ";
                std::cout << std::endl;
            }
            if(outputSize > 5) std::cout << " ... " << std::endl;
        }
        if(outputPlanes > 5) std::cout << "   ... other planes ... " << std::endl;
    }
    if(batchSize > 5) std::cout << "   ... other n ... " << std::endl;
}
VIRTUAL void InputLayer::print() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: print");
#endif


    printOutput();
}
 void InputLayer::in(float const*images) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: in");
#endif


//        std::cout << "InputLayer::in()" << std::endl;
    this->input = images;

//        this->batchStart = batchStart;
//        this->batchEnd = batchEnd;
//        print();
}
VIRTUAL bool InputLayer::needErrorsBackprop() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: needErrorsBackprop");
#endif


    return false;
}
VIRTUAL void InputLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: setBatchSize");
#endif


//        std::cout << "inputlayer setting batchsize " << batchSize << std::endl;

#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\layer\\InputLayer.cpp: InputLayer setBatchSize");
#endif

    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
//    if(output != 0) {
//        delete[] output;
//    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
//    output = new float[batchSize * getOutputCubeSize() ];
    if (not setup){
    	//LOGI( "DeepCL/src/input/InputLayer.cpp: not setup");
		float * test=0;
		#if EIGHT_BIT_ACCURACY==1
			unsigned char* test3;
			outputWrapper8Bit=(CLWrapper*) cl->wrap8Bit(batchSize * getOutputCubeSize() , test3);
			outputWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
		#else
		#if HALF_ACCURACY==0
				outputWrapper= cl->wrap(batchSize * getOutputCubeSize() , test);
				outputWrapper->createZeroCopyObject_WriteFlag_OnDevice();
		#else
				half * test2=0;
				outputWrapperHalf= (CLWrapper*) cl->wrap(batchSize * getOutputCubeSize() , test2);
				outputWrapperHalf->createZeroCopyObject_WriteFlag_OnDevice();
		#endif
		#endif
    	setup=true;

    }
//    const float * test;
//    outputWrapper= cl->wrap(batchSize * getOutputCubeSize() , test);
//    outputWrapper->createOnDevice();


}

std::string to_string_with_precisionI(const float a_value, const int n = 2)
{
	std::stringstream ss;
	if (a_value==0)
		ss << std::fixed << 0;
	else
		ss << std::fixed << std::setprecision(n) << a_value;
    return ss.str();
}

void InputLayer::encode8BitWeight(const float * array,int len, unsigned char *dst,float &minW,float &multW){

	    float min = 999.99, max = -999.99, mult;
		int i;
		for(i = 0; i < len; i++)
		{
			if(array[i] < min)
				min = array[i];
			if(array[i] > max)
				max = array[i];
		}
		if(max - min > 0)
			mult = 255.0 / (max - min);
		else mult = 0;
//		unsigned char *dst = new unsigned char[len];
		for(i = 0; i < len; i++)
			dst[i] = roundf(array[i] * mult-min*mult);

		minW=min;
		multW=mult;
//		for(int i=0; i<10;i++){
//				LOGI("dst[%d]=%i",i,dst[i]);
//			}
//
		LOGI("min=%f mult=%f max %f",min,mult,max);

}

VIRTUAL void InputLayer::forward() {
#if 1//TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: forward");
#endif

#if EIGHT_BIT_ACCURACY==1
	unsigned char* test3;
	unsigned char *dst = new unsigned char[batchSize * getOutputCubeSize()];
	encode8BitWeight(input,batchSize * getOutputCubeSize(),dst, min,mult);
	test3=outputWrapper8Bit->map_ZeroCopyObject_WriteFlag8Bit();

	for(int i=0;i<batchSize * getOutputCubeSize();i++){
		test3[i]=dst[i];
	}
	outputWrapper8Bit->unMap_ZeroCopyObject_WriteFlag8Bit(test3);
	delete[] dst;

//	unsigned char* temp;
//	temp=outputWrapper8Bit->map_ZeroCopyObject_ReadFlag8Bit();
//	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
//		LOGI("--------------------RAW output8Bit[%d]=%f ",i,((float)temp[i])*mult+min);
//	}
//	outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlag8Bit(temp);

#else

	#if HALF_ACCURACY==0
		outputWrapper->copyToDevice_ZeroCopyObject_ReadWriteFlag(input);
	#else
		outputWrapperHalf->copyToDevice_ZeroCopyObject_ReadWriteFlagHalf(input);
	#endif
#endif

#if 0
	half *test =outputWrapperHalf->map_ZeroCopyObject_ReadFlagHalf();
	float e=0;
	for(int i=0; i<outputWrapper->size();i++){
		e=input[i]-HalfToFloat(test[i]);
	}
	LOGI("error=%f",e);
	outputWrapperHalf->unMap_ZeroCopyObject_ReadFlagHalf(test);
#endif

}
//VIRTUAL void InputLayer::backward(float learningRate, float const *gradOutput) {
//}
VIRTUAL int InputLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL int InputLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: getOutputPlanes");
#endif


    return outputPlanes;
}
VIRTUAL int InputLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: getOutputCubeSize");
#endif


    return outputPlanes * outputSize * outputSize;
}
VIRTUAL int InputLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: getOutputNumElements");
#endif


    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string InputLayer::toString() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: string InputLayer::toString");
#endif


    return asString();
}
VIRTUAL std::string InputLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: string InputLayer::asString");
#endif


    return std::string("") + "InputLayer{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayer.cpp: string");
#endif


}

VIRTUAL bool InputLayer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: hasOutputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper * InputLayer::getOutput8bitInfo(float &minW,float &multW){
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutput8bitInfo");
#endif
	minW=min;
	multW=mult;
	return outputWrapper8Bit;

}

VIRTUAL CLWrapper *InputLayer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputWrapper");
#endif


    return outputWrapper;
}

VIRTUAL CLWrapper *InputLayer::getOutputWrapperTest() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputWrapper");
#endif


    return outputWrapperHalf;
}

//template<>VIRTUAL std::string InputLayer<unsigned char>::asString() const {
//    return std::string("") + "InputLayer<unsigned char>{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
//}

//template<>VIRTUAL std::string InputLayer<float>::asString() const {
//    return std::string("") + "InputLayer<float>{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
//}


