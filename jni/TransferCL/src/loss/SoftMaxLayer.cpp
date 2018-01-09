// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "../../EasyCL/util/StatefulTimer.h"

#include "../layer/LayerMaker.h"
#include "../fc/FullyConnectedLayer.h"
#include "SoftMaxLayer.h"

using namespace std;

#define TEST_SOFTMAX 0
#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayer::SoftMaxLayer(Layer *previousLayer, SoftMaxMaker *maker,int batch) :
    LossLayer(previousLayer, maker),
        perPlane(maker->_perPlane),
        ptrcl(maker->cl),
        imageSize(previousLayer->getOutputSize()),
        numPlanes(previousLayer->getOutputPlanes()),
        imageSizeSquared(previousLayer->getOutputSize() * previousLayer->getOutputSize()),
        output(0),
        gradInput(0),
        allocatedSize(0),
        kernel(0),
        setup1(false)
         {

    batchSize=batch;
	prediction=false;
	setup=false;
	#if HALF_ACCURACY==0
		output1 = new float [batchSize*numPlanes];
	#endif
}
VIRTUAL SoftMaxLayer::~SoftMaxLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: ~SoftMaxLayer");
#endif

	#if HALF_ACCURACY==0
		if(gradInput != 0) {
			delete[] gradInput;
		}

		delete[] output1;

		if (kernel!= 0) {
			delete kernel;
		}
		//delete outputWrapper;
		if (setup){
			delete[] lossFloat;
			delete lossWrapper;
			delete[] nbRight;
			delete 	nbRightWrapper;
			delete[] labelData;
			delete labelWrapper;
			delete gradInputWrapper;
		}
	#else
		if(ptrcl->training){
			delete lossWrapperHalf;
			delete nbRightWrapperHalf;
			delete labelWrapperHalf;
			delete gradInputWrapperHalf;
		}

#if 0
		delete[] output;
		delete[] gradInput;
#endif

	#endif
}
VIRTUAL std::string SoftMaxLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: string SoftMaxLayer::getClassName");
#endif


    return "SoftMaxLayer";
}
VIRTUAL float *SoftMaxLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getOutput");
#endif


// olivier the operation is performed by calcLossFromLabels
ptrcl->finish();
float *input=previousLayer->getOutputWrapper()->map_ZeroCopyObject_ReadFlag();


//previousLayer->getOutputWrapper()->copyToHost();
//float *input = (float *)(previousLayer->getOutputWrapper()->getHostArray());//previousLayer->getOutput(); // just retrieve as host-side array for now
//ptrcl->finish();

if (not prediction){
	if(perPlane) {
		   for(int n = 0; n < batchSize; n++) {
			   for(int plane = 0; plane < numPlanes; plane++) {
				   int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
				   float maxValue = input[imageOffset + 0];
				   for(int i = 1; i < imageSizeSquared; i++) {
					   maxValue = std::max(maxValue, input[imageOffset + i]);
				   }
				   float denominator = 0;
				   for(int i = 0; i < imageSizeSquared; i++) {
					   denominator += exp(input[imageOffset + i] - maxValue);
				   }
				   for(int i = 0; i < imageSizeSquared; i++) {
					   output1[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
				   }
			   }
		   }
	   } else {
		   // force imagesize of 1 for now
		   if(imageSize != 1) {
			   throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
		   }
		   for(int n = 0; n < batchSize; n++) {
			   int imageOffset = n * numPlanes * imageSizeSquared;
			   // first get the max
			   float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
			   for(int plane = 1; plane < numPlanes; plane++) {
				   maxValue = std::max(maxValue, input[imageOffset + plane]);
			   }
			   // calculate sum, under this max
			   float denominator = 0;
			   for(int plane = 0; plane < numPlanes; plane++) {
				   denominator += exp(input[imageOffset + plane] - maxValue);
			   }
			   // now calc the softmaxes:
			   for(int plane = 0; plane < numPlanes; plane++) {
				   output1[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
			   }

		}
	   }
	//prediction=true;//to comment
	}
	previousLayer->getOutputWrapper()->unMap_ZeroCopyObject_ReadFlag(input);

    return output1;
}
VIRTUAL float *SoftMaxLayer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getGradInput");
#endif


    return gradInput;
}
VIRTUAL void SoftMaxLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: setBatchSize");
#endif

    this->batchSize = batchSize;
    if(batchSize <= this->allocatedSize) {
        return;
    }
//    if(output != 0) {
//        delete[] output;
//    }
	if (not setup1){
#if 0
	    output = new float[ getOutputNumElements() ];
	    gradInput = new float[ previousLayer-> getOutputNumElements() ];
#endif

		allocatedSize = batchSize;
		setup1=true;
	}

}
VIRTUAL int SoftMaxLayer::getBatchSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getBatchSize");
#endif


    return this->batchSize;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLossFromLabels(int const *labels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: calcLossFromLabels %d",perPlane);
#endif


	if (not setup){
		#if HALF_ACCURACY==0
			float* temp1;
			lossFloat=new float[1];
			lossWrapper = ptrcl->wrap(1,lossFloat);
			lossWrapper->createOnDevice();
			lossWrapper->copyToDevice();
			nbRight=new int[1];
			nbRightWrapper = ptrcl->wrap(1,nbRight);
			nbRightWrapper->createOnDevice();
			nbRightWrapper->copyToDevice();
			labelData =new int[batchSize];
			labelWrapper = ptrcl->wrap(batchSize,labelData);
			labelWrapper->createOnDevice();
			gradInputWrapper= (CLWrapper*) ptrcl->wrap(previousLayer-> getOutputNumElements(),temp1);
			gradInputWrapper->createZeroCopyObject_ReadWriteFlag_OnDevice();//createZeroCopyObject_HostNotAccessFlag_OnDevice();
	//
	//		gradInputWrapper=ptrcl->wrap(previousLayer-> getOutputNumElements(),gradInput);
	//		gradInputWrapper->createOnDevice();
			globalSize = batchSize;
			workgroupsize = std::min(globalSize, ptrcl->getMaxWorkgroupSize());
			createKernel();
			setup=true;
			#if EIGHT_BIT_ACCURACY==1
				float noRelevant1;
				float noRelevant2;
				inputWrapper = previousLayer->getOutput8bitInfo(noRelevant1,noRelevant2);
			#else
				inputWrapper =  previousLayer->getOutputWrapper();
			#endif
			//inputWrapper = previousLayer->getOutputWrapper();

			kernel->input(inputWrapper);
			//kernel->output(outputWrapper);
			kernel->input(labelWrapper);
			kernel->output(lossWrapper);
			kernel->output(nbRightWrapper);
			kernel->output(gradInputWrapper);

		#else
			globalSize = batchSize;
			workgroupsize = std::min(globalSize, ptrcl->getMaxWorkgroupSize());
			int* temp2;
			half* temp3;
			lossWrapperHalf= (CLWrapper*) ptrcl->wrap(1,temp3);
			lossWrapperHalf->createZeroCopyObject_ReadFlag_OnDevice();
			nbRightWrapperHalf= (CLWrapper*) ptrcl->wrap(1,temp2);
			nbRightWrapperHalf->createZeroCopyObject_ReadFlag_OnDevice();
			labelWrapperHalf= (CLWrapper*) ptrcl->wrap(batchSize,temp2);
			labelWrapperHalf->createZeroCopyObject_WriteFlag_OnDevice();
			gradInputWrapperHalf= (CLWrapper*) ptrcl->wrap(previousLayer-> getOutputNumElements(),temp3);
			gradInputWrapperHalf->createZeroCopyObject_HostNotAccessFlag_OnDevice();

			createKernelHalf();
			#if EIGHT_BIT_ACCURACY==1
				float noRelevant1;
				float noRelevant2;
				inputWrapperHalf = previousLayer->getOutput8bitInfo(noRelevant1,noRelevant2);
			#else
				inputWrapperHalf = previousLayer->getOutputWrapperTest();
			#endif

	//		float *weightstemp=inputWrapper->map_ZeroCopyObject_ReadFlag();
	//		inputWrapperHalf->copyToDevice_ZeroCopyObject_WriteFlagHalf(weightstemp);
	//		inputWrapper->unMap_ZeroCopyObject_ReadFlag(weightstemp);

			kernelHalf->input(inputWrapperHalf);
			//kernel->output(outputWrapper);
			kernelHalf->input(labelWrapperHalf);
			kernelHalf->output(lossWrapperHalf);
			kernelHalf->output(nbRightWrapperHalf);
			kernelHalf->output(gradInputWrapperHalf);
		#endif
		setup=true;
	}


//	inputWrapper = previousLayer->getOutputWrapper();
#if HALF_ACCURACY==0
	runSoftmax_forward( inputWrapper/*,  outputWrapper*/,labels) ;
#else
	runSoftmax_forwardHalf( inputWrapperHalf/*,  outputWrapper*/,labels) ;
#endif
	////////////////////////////

#if 0
			////////////////////////////
			float *testPrev= gradInputWrapper->map_ZeroCopyObject_ReadFlag();
			half* testPrevHalf=gradInputWrapperHalf->map_ZeroCopyObject_ReadFlagHalf();
		float e=0;
			for(int i = 0; i<previousLayer-> getOutputNumElements();i++){
				e=e+abs(testPrev[i]-HalfToFloat(testPrevHalf[i]));
			}
//		//
//			for(int i = 300; i< 400/*previousLayer->getOutputNumElements()*/;i++){
//		//		if (abs(testPrev[i]-HalfToFloat(testPrevHalf[i]))>0.1)
//					LOGI("%d) %f vs %f nb = %d %f",i, testPrev[i],HalfToFloat(testPrevHalf[i]),previousLayer->getOutputNumElements(),abs(testPrev[i]-HalfToFloat(testPrevHalf[i])));
//				}

			LOGI(" e");
			LOGI("softmax error %f",e);

			gradInputWrapperHalf->unMap_ZeroCopyObject_ReadFlagHalf(testPrevHalf);
			gradInputWrapper->unMap_ZeroCopyObject_ReadFlag(testPrev);

			/////////////////////////////

	LOGI("#########################\n\n\n");

	for(int i=0;i<20;i++){
		LOGI("output[%d]=%f",i,output[ i ]);
	}

	    if(perPlane) {
	        for(int n = 0; n < batchSize; n++) {
	            for(int plane = 0; plane < numPlanes; plane++) {
	                int label = labels[n * numPlanes + plane];
	                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
	                lossToDelete += - log(std::max(output[ imageOffset + label ], FLT_MIN));
	            }
	        }
	    } else {
	        // force imagesize of 1 for now
	        if(imageSize != 1) {
	            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
	        }
	        for(int n = 0; n < batchSize; n++) {
	            int imageOffset = n * numPlanes * imageSizeSquared;
	            int label = labels[n];
	            lossToDelete += - log(std::max(output[imageOffset + label], FLT_MIN));
	        }
	    }
	    LOGI("#########################\n\n\n");
	    if(perPlane) {
	        for(int n = 0; n < batchSize; n++) {
	            for(int plane = 0; plane < numPlanes; plane++) {
	                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
	                int label = labels[n * numPlanes + plane];

	                for(int i = 0; i < imageSizeSquared; i++) {
	                    float value = output[imageOffset + i];
	                    if (std::isfinite(value) == false)
	                        throw runtime_error("Output is a non-finite number, this usually means the learning rate is too high");
	                    gradInput[imageOffset + i] = value;
	                }
	                gradInput[imageOffset + label] -= 1;
	            }
	        }
	    } else {
	        // force imagesize of 1 for now
	        if(imageSize != 1) {
	            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
	        }
	        for(int n = 0; n < batchSize; n++) {
	            int imageOffset = n * numPlanes * imageSizeSquared;
	            int label = labels[n];
	            for(int plane = 0; plane < numPlanes; plane++) {
	                float value = output[imageOffset + plane];
	                if (std::isfinite(value) == false)
	                    throw runtime_error("Output is a non-finite number, this usually means the learning rate is too high");
	                gradInput[imageOffset + plane] = value;
	            }
	            if(label >= numPlanes) {
	                throw runtime_error("Label " + toString(label) + " exceeds number of softmax planes " + toString(numPlanes) );
	            } else if(label < 0) {
	                throw runtime_error("Label " + toString(label) + " negative");
	            }
	            gradInput[imageOffset + label] -= 1;
	        }
	    }

	    half *testPrev2= gradInputWrapperHalf->map_ZeroCopyObject_ReadFlagHalf();

	    for(int i=0; i<100;i++){
	    	LOGI("gradInputWrapper[%d] %f vs %f",i, gradInput[i],HalfToFloat(testPrev2[i]));
	    }
	    LOGI("gradInputWrapperHalf->size() %d",gradInputWrapperHalf->size());
	    e=0;
	    	for(int i = 0; i<previousLayer-> getOutputNumElements();i++){
	    		e=e+abs(gradInput[i]-HalfToFloat(testPrev2[i]));
	    	}
	    	LOGI("gradInputWrapper error %f",e);

	    	gradInputWrapperHalf->unMap_ZeroCopyObject_ReadFlagHalf(testPrev2);



#endif

    return lossFloat[0];
}

VIRTUAL bool SoftMaxLayer::providesGradInputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
	LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: providesGradInputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *SoftMaxLayer::getGradInputWrapper() {
#if TRANSFERCL_VERBOSE == 1
	LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapper;
}

VIRTUAL CLWrapper *SoftMaxLayer::getGradInputWrapperTest() {
#if TRANSFERCL_VERBOSE == 1
	LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapperHalf;
}

// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLoss(float const *expectedValues) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "-------------ERROR --->  DeepCL/src/loss/SoftMaxLayer.cpp: calcLoss");
#endif


    StatefulTimer::timeCheck("start SoftMaxLayer calcLoss");
    float loss = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    if(expectedValues[ imageOffset + i ] != 0) {
                        float thisloss = - expectedValues[ imageOffset + i ] * log(output[ imageOffset + i ]);
                        loss += thisloss;
                    }
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                float thisloss = - expectedValues[imageOffset + plane] * log(output[imageOffset + plane]);
                loss += thisloss;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLoss");
    return loss;
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcGradInputFromLabels(int const *labels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: calcGradInputFromLabels");
#endif

//olivier:this is the CPU version. Everything is already done on the GPU (runSoftmax_forward)
//	float *temp =new float [batchSize*numPlanes];
//	for (int i =0; i<batchSize*numPlanes;i++)
//		temp[i]=gradInput[i];
//
////    cout << "softmaxlayer::calcerrors" << endl;
//    StatefulTimer::timeCheck("start SoftMaxLayer calcGradInputfromlabels");
//    if(perPlane) {
//        for(int n = 0; n < batchSize; n++) {
//            for(int plane = 0; plane < numPlanes; plane++) {
//                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
//                int label = labels[n * numPlanes + plane];
//                for(int i = 0; i < imageSizeSquared; i++) {
//                    gradInput[imageOffset + i] = output[imageOffset + i];
//                }
//                gradInput[imageOffset + label] -= 1;
//            }
//        }
//    } else {
//        // force imagesize of 1 for now
//        if(imageSize != 1) {
//            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
//        }
//        for(int n = 0; n < batchSize; n++) {
//            int imageOffset = n * numPlanes * imageSizeSquared;
//            int label = labels[n];
//            for(int plane = 0; plane < numPlanes; plane++) {
//                gradInput[imageOffset + plane] = output[imageOffset + plane];
//            }
//            if(label >= numPlanes) {
//                throw runtime_error("Label " + toString(label) + " exceeds number of softmax planes " + toString(numPlanes) );
//            } else if(label < 0) {
//                throw runtime_error("Label " + toString(label) + " negative");
//            }
//            gradInput[imageOffset + label] -= 1;
//        }
//    }
//    StatefulTimer::timeCheck("end SoftMaxLayer calcGradInputfromlabels");
//    float error =0.0f;
//	for (int i =0; i<batchSize*numPlanes;i++)
//		error=abs(temp[i]-gradInput[i]);
//
//delete[] temp;
//	  LOGI("error=%f",error);
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcGradInput(float const *expectedValues) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "-------------ERROR --->  DeepCL/src/loss/SoftMaxLayer.cpp: calcGradInput");
#endif


//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcGradInput");
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    int resultIndex = imageOffset + i;
                    gradInput[resultIndex] = output[resultIndex] - expectedValues[resultIndex];
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                int resultIndex = imageOffset + plane;
                gradInput[resultIndex] = output[resultIndex] - expectedValues[resultIndex];
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcGradInput");
}
VIRTUAL int SoftMaxLayer::getNumLabelsPerExample() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getNumLabelsPerExample");
#endif


    if(perPlane) {
        return numPlanes;
    } else {
        return imageSizeSquared;
    }
}
VIRTUAL int SoftMaxLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL int SoftMaxLayer::calcNumRightFromLabels(int const*labels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: calcNumRightFromLabels");
#endif

int numRight = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                int label = labels[n * numPlanes + plane];
                float thisMax = output[imageOffset + 0];
                int iMax = 0;
                for(int i = 1; i < imageSizeSquared; i++) {
                    if(output[imageOffset + i] > thisMax) {
                        thisMax = output[imageOffset + i];
                        iMax = i;
                    }
                }
                if(label == iMax) {
//                    cout << "n " << n << " plane " << plane << " label " << label << endl;
                    numRight++;
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            float thisMax = output[imageOffset + 0];
            int iMax = 0;
            for(int i = 1; i < numPlanes; i++) {
                if(output[imageOffset + i] > thisMax) {
                    thisMax = output[imageOffset + i];
                    iMax = i;
                }
            }
            if(label == iMax) {
                numRight++;
            }
        }
    }

    LOGI("%d vs %d",nbRight[0],numRight);


    return nbRight[0];
}
// for forward, we just need to apply the softmax activation. "just" :-P
VIRTUAL void SoftMaxLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: forward perPlane %d %d %d",perPlane,imageSizeSquared,numPlanes);
#endif
#if 0
	float * foo;
	foo = new float [batchSize*numPlanes];




	// olivier the operation is performed by calcLossFromLabels
//	ptrcl->finish();
//	previousLayer->getOutputWrapper()->copyToHost();
//	float *input = (float *)(previousLayer->getOutputWrapper()->getHostArray());//previousLayer->getOutput(); // just retrieve as host-side array for now
//	ptrcl->finish();
#if 0
	float *input = (float *)(previousLayer->getOutputWrapper()->map_ZeroCopyObject_ReadFlag());
#else
	half *inputh = (half *)(previousLayer->getOutputWrapperTest()->map_ZeroCopyObject_ReadFlag());
	float *input=new float[previousLayer->getOutputWrapperTest()->size()];
	for(int i=0;i<previousLayer->getOutputWrapperTest()->size();i++){
		input[i]=HalfToFloat(inputh[i]);
	}
#endif

	if(perPlane) {
		   for(int n = 0; n < batchSize; n++) {
			   for(int plane = 0; plane < numPlanes; plane++) {
				   int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
				   float maxValue = input[imageOffset + 0];
				   for(int i = 1; i < imageSizeSquared; i++) {
					   maxValue = std::max(maxValue, input[imageOffset + i]);
				   }
				   float denominator = 0;
				   for(int i = 0; i < imageSizeSquared; i++) {
					   denominator += exp(input[imageOffset + i] - maxValue);
				   }
				   for(int i = 0; i < imageSizeSquared; i++) {
					   output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
				   }
			   }
		   }
	   } else {
		   // force imagesize of 1 for now
		   if(imageSize != 1) {
			   throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
		   }
		   for(int n = 0; n < batchSize; n++) {
			   int imageOffset = n * numPlanes * imageSizeSquared;
			   // first get the max
			   float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
			   for(int plane = 1; plane < numPlanes; plane++) {
				   maxValue = std::max(maxValue, input[imageOffset + plane]);
			   }
			   // calculate sum, under this max
			   float denominator = 0;
			   for(int plane = 0; plane < numPlanes; plane++) {
				   denominator += exp(input[imageOffset + plane] - maxValue);
			   }
			   // now calc the softmaxes:
			   for(int plane = 0; plane < numPlanes; plane++) {
				   output[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
			   }
	//           maxValue=0.0;
	//		   int classId=-1;
	//		   for(int plane = 1; plane < numPlanes; plane++) {
	//			   if(maxValue<foo[imageOffset + plane]) {
	//					maxValue = std::max(maxValue, foo[imageOffset + plane]);
	//					classId=plane;
	//			   }
	//		   }
	//		   if (imageOffset<3)
	//			   LOGI("class %d",classId);
		   }
	   }
	LOGI("calc");
	for(int n = 0; n < 4; n++) {
			float *outputStack = output + n * numPlanes;
			float highestProb = outputStack[0];
			int bestPlane = 0;
			for(int plane = 1; plane < numPlanes; plane++) {
				if(outputStack[plane] > highestProb) {
					bestPlane = plane;
					highestProb = outputStack[plane];
				}
			}
			LOGI("bestPlane[%d]=%d (%f)",n,bestPlane,highestProb);
			//labels[n] = bestPlane;
		}

#if 0
	previousLayer->getOutputWrapper()->unMap_ZeroCopyObject_ReadFlag(input);
#else
	delete[] input;
	previousLayer->getOutputWrapperTest()->unMap_ZeroCopyObject_ReadFlagHalf(inputh);
#endif

	delete[] foo;

    StatefulTimer::timeCheck("end SoftMaxLayer forward");
#endif
}
VIRTUAL void SoftMaxLayer::getLabels(int *labels) { // need to allocate labels array first, and have called 'forward' first
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getLabels");
#endif


	float * foo;
	foo = new float [batchSize*numPlanes];

	// olivier the operation is performed by calcLossFromLabels
	ptrcl->finish();
#if	EIGHT_BIT_ACCURACY==0
	#if HALF_ACCURACY==0
		float *input=previousLayer->getOutputWrapper()->map_ZeroCopyObject_ReadFlag();


		if(perPlane) {
			   for(int n = 0; n < batchSize; n++) {
				   for(int plane = 0; plane < numPlanes; plane++) {
					   int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
					   float maxValue = input[imageOffset + 0];
					   for(int i = 1; i < imageSizeSquared; i++) {
						   maxValue = std::max(maxValue, input[imageOffset + i]);
					   }
					   float denominator = 0;
					   for(int i = 0; i < imageSizeSquared; i++) {
						   denominator += exp(input[imageOffset + i] - maxValue);
					   }
					   for(int i = 0; i < imageSizeSquared; i++) {
						   output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
					   }
				   }
			   }
		   } else {

			   if(imageSize != 1) {
				   throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
			   }
			   for(int n = 0; n < batchSize; n++) {
				   int imageOffset = n * numPlanes * imageSizeSquared;
				   // first get the max
				   float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
				   for(int plane = 1; plane < numPlanes; plane++) {
					   maxValue = std::max(maxValue, input[imageOffset + plane]);
				   }
				   // calculate sum, under this max
				   float denominator = 0;
				   for(int plane = 0; plane < numPlanes; plane++) {
					   denominator += exp(input[imageOffset + plane] - maxValue);
				   }
				   // now calc the softmaxes:
				   for(int plane = 0; plane < numPlanes; plane++) {
					   foo[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
				   }

				float *outputStack = foo + n * numPlanes;
				float highestProb = outputStack[0];
				LOGI("plane %d prob = %f",0,outputStack[0]);
				int bestPlane = 0;
				for(int plane = 1; plane < numPlanes; plane++) {
					LOGI("plane %d prob = %f %f %d",plane,outputStack[plane],highestProb,bestPlane);
					if(outputStack[plane] > highestProb) {
						bestPlane = plane;
						highestProb = outputStack[plane];
					}
				}

				labels[n] = bestPlane;

			 }

		   }

			previousLayer->getOutputWrapper()->unMap_ZeroCopyObject_ReadFlag(input);



			delete[] foo;

	#else
			half *testPrev= previousLayer->getOutputWrapperTest()->map_ZeroCopyObject_ReadFlagHalf();
			float *input= new float[previousLayer->getOutputWrapperTest()->size()];
			ptrcl->finish();
			for(int i=0; i<previousLayer->getOutputWrapperTest()->size();i++){
				input[i]=HalfToFloat(testPrev[i]);
				//LOGI("input %f",input[i]);
			}

			if(perPlane) {
				   for(int n = 0; n < batchSize; n++) {
					   for(int plane = 0; plane < numPlanes; plane++) {
						   int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
						   float maxValue = input[imageOffset + 0];
						   for(int i = 1; i < imageSizeSquared; i++) {
							   maxValue = std::max(maxValue, input[imageOffset + i]);
						   }
						   float denominator = 0;
						   for(int i = 0; i < imageSizeSquared; i++) {
							   denominator += exp(input[imageOffset + i] - maxValue);
						   }
						   for(int i = 0; i < imageSizeSquared; i++) {
							   output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
						   }
					   }
				   }
			   } else {

				   if(imageSize != 1) {
					   throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
				   }
				   for(int n = 0; n < batchSize; n++) {
					   int imageOffset = n * numPlanes * imageSizeSquared;
					   // first get the max
					   float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
					   for(int plane = 1; plane < numPlanes; plane++) {
						   maxValue = std::max(maxValue, input[imageOffset + plane]);
					   }
					   // calculate sum, under this max
					   float denominator = 0;
					   for(int plane = 0; plane < numPlanes; plane++) {
						   denominator += exp(input[imageOffset + plane] - maxValue);
					   }
					   // now calc the softmaxes:
					   for(int plane = 0; plane < numPlanes; plane++) {
						   foo[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
					   }

					float *outputStack = foo + n * numPlanes;
					float highestProb = outputStack[0];
					int bestPlane = 0;
					for(int plane = 1; plane < numPlanes; plane++) {
						if(outputStack[plane] > highestProb) {
							bestPlane = plane;
							highestProb = outputStack[plane];
						}
					}
	//		        LOGI("bestPlane %d",bestPlane);
					labels[n] = bestPlane;

				 }

			   }

			   previousLayer->getOutputWrapperTest()->unMap_ZeroCopyObject_ReadFlagHalf(testPrev);


				delete[] input;
				delete[] foo;

	#endif
#else
	#if HALF_ACCURACY==0
		float noRelevant1;
		float noRelevant2;
		float *input=previousLayer->getOutput8bitInfo(noRelevant1,noRelevant2)->map_ZeroCopyObject_ReadFlag();


		if(perPlane) {
			   for(int n = 0; n < batchSize; n++) {
				   for(int plane = 0; plane < numPlanes; plane++) {
					   int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
					   float maxValue = input[imageOffset + 0];
					   for(int i = 1; i < imageSizeSquared; i++) {
						   maxValue = std::max(maxValue, input[imageOffset + i]);
					   }
					   float denominator = 0;
					   for(int i = 0; i < imageSizeSquared; i++) {
						   denominator += exp(input[imageOffset + i] - maxValue);
					   }
					   for(int i = 0; i < imageSizeSquared; i++) {
						   output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
					   }
				   }
			   }
		   } else {

			   if(imageSize != 1) {
				   throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
			   }
			   for(int n = 0; n < batchSize; n++) {
				   int imageOffset = n * numPlanes * imageSizeSquared;
				   // first get the max
				   float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
				   for(int plane = 1; plane < numPlanes; plane++) {
					   maxValue = std::max(maxValue, input[imageOffset + plane]);
				   }
				   // calculate sum, under this max
				   float denominator = 0;
				   for(int plane = 0; plane < numPlanes; plane++) {
					   denominator += exp(input[imageOffset + plane] - maxValue);
				   }
				   // now calc the softmaxes:
				   for(int plane = 0; plane < numPlanes; plane++) {
					   foo[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
				   }

				float *outputStack = foo + n * numPlanes;
				float highestProb = outputStack[0];
				LOGI("plane %d prob = %f",0,outputStack[0]);
				int bestPlane = 0;
				for(int plane = 1; plane < numPlanes; plane++) {
					LOGI("plane %d prob = %f %f %d",plane,outputStack[plane],highestProb,bestPlane);
					if(outputStack[plane] > highestProb) {
						bestPlane = plane;
						highestProb = outputStack[plane];
					}
				}

				labels[n] = bestPlane;

			 }

		   }

			previousLayer->getOutput8bitInfo(noRelevant1,noRelevant2)->unMap_ZeroCopyObject_ReadFlag(input);



			delete[] foo;

	#else
			float noRelevant1;
			float noRelevant2;
			ptrcl->finish();
			half *testPrev= previousLayer->getOutput8bitInfo(noRelevant1,noRelevant2)->map_ZeroCopyObject_ReadFlagHalf();
			float *input= new float[previousLayer->getOutputNumElements()];

			for(int i=0; i<previousLayer->getOutputNumElements();i++){
				input[i]=HalfToFloat(testPrev[i]);
			}

			if(perPlane) {
				   for(int n = 0; n < batchSize; n++) {
					   for(int plane = 0; plane < numPlanes; plane++) {
						   int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
						   float maxValue = input[imageOffset + 0];
						   for(int i = 1; i < imageSizeSquared; i++) {
							   maxValue = std::max(maxValue, input[imageOffset + i]);
						   }
						   float denominator = 0;
						   for(int i = 0; i < imageSizeSquared; i++) {
							   denominator += exp(input[imageOffset + i] - maxValue);
						   }
						   for(int i = 0; i < imageSizeSquared; i++) {
							   output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
						   }
					   }
				   }
			   } else {

				   if(imageSize != 1) {
					   throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
				   }
				   for(int n = 0; n < batchSize; n++) {
					   int imageOffset = n * numPlanes * imageSizeSquared;
					   // first get the max
					   float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
					   for(int plane = 1; plane < numPlanes; plane++) {
						   maxValue = std::max(maxValue, input[imageOffset + plane]);
					   }
					   // calculate sum, under this max
					   float denominator = 0;
					   for(int plane = 0; plane < numPlanes; plane++) {
						   denominator += exp(input[imageOffset + plane] - maxValue);
					   }
					   // now calc the softmaxes:
					   for(int plane = 0; plane < numPlanes; plane++) {
						   foo[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
					   }

					float *outputStack = foo + n * numPlanes;
					float highestProb = outputStack[0];
					int bestPlane = 0;
					for(int plane = 1; plane < numPlanes; plane++) {
						if(outputStack[plane] > highestProb) {
							bestPlane = plane;
							highestProb = outputStack[plane];
						}
					}
	//		        LOGI("bestPlane %d",bestPlane);
					labels[n] = bestPlane;

				 }

			   }

			   previousLayer->getOutput8bitInfo(noRelevant1,noRelevant2)->unMap_ZeroCopyObject_ReadFlagHalf(testPrev);


				delete[] input;
				delete[] foo;

	#endif
#endif
}

VIRTUAL std::string SoftMaxLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: string SoftMaxLayer::asString");
#endif


    return "SoftMaxLayer{ perPlane=" + toString(perPlane) + " numPlanes=" + toString(numPlanes)
        + " imageSize=" + toString(imageSize) + " }";
}

void SoftMaxLayer::createKernel(){

	buildKernelSoftmax_forward();

}

void SoftMaxLayer::buildKernelSoftmax_forward() {
    TemplatedKernel builder(ptrcl);


        setupBuilderSoftmax_forward(&builder);

        string identifier2="softmax_forward"+std::to_string(numPlanes);
        identifier2=identifier2+"_InputSize="+std::to_string(imageSize);
        identifier2=identifier2+"_batchsize="+std::to_string(batchSize);
        string test=getKernelTemplateSoftmax_forward();
        this->kernel = builder.buildKernel(
           		identifier2,
               "softmax",
               test,
               "softmax_forward",
               false
        );
    }

void SoftMaxLayer::runSoftmax_forward(    CLWrapper *inputWrapper/*,CLWrapper *outputWrapper*/,int const *labels) {

#if TEST_SOFTMAX==1
clock_t startTimer1, stopTimer1,startTimer2, stopTimer2,startTimer3, stopTimer3;
	startTimer2=clock();
#endif

	labelWrapper->copyToDevice(labels);



#if TEST_SOFTMAX==1
	LOGI("globalSize %d workgroupsize %d",globalSize,workgroupsize);
	startTimer1=clock();
#endif

	kernel->run_1d(globalSize, workgroupsize);
	//ptrcl->finish();
#if COMPAREHALF==1
	testgrad =new float[previousLayer-> getOutputNumElements()];
	gradInputWrapper->copyToHost(testgrad);
	for(int i=0;i<10;i++){
		LOGI("grad[%d]=%f",i,testgrad[i]);
	}
#endif
	#if TEST_SOFTMAX==1
	stopTimer1 = clock();
	startTimer3=clock();
	#endif

	#if TEST_SOFTMAX==1

		stopTimer3 = clock();
		LOGI("read  took %g ms\n\n", 1000.0* (double)(stopTimer3 - startTimer3)/(double)CLOCKS_PER_SEC) ;
		LOGI("Softmax  took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		LOGI("loss %f", lossFloat[0]) ;

		stopTimer2 = clock();
		LOGI("Softmax  took %g ms\n\n", 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC) ;
	#endif


}

CLWrapper * SoftMaxLayer::getLossWrapper(){
	#if HALF_ACCURACY==0
		return lossWrapper;
	#else
		return lossWrapperHalf;
	#endif
}

CLWrapper * SoftMaxLayer::getNbRightWrapper(){
	#if HALF_ACCURACY==0
		return nbRightWrapper;
	#else
		return nbRightWrapperHalf;
	#endif
}

void SoftMaxLayer::setupBuilderSoftmax_forward(TemplatedKernel *builder) {

	int possibleGlobalSize =batchSize;// batchSize;//crash because batchSize value = 0
	int possibleWorkgroupsize = std::min(possibleGlobalSize, ptrcl->getMaxWorkgroupSize());
	string hintCompilerString="__attribute__((vec_type_hint(";
	hintCompilerString+="float";
	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);


	builder->set("numPlanes",numPlanes);
	builder->set("batchSize",batchSize);

}

STATIC std::string SoftMaxLayer::getKernelTemplateSoftmax_forward() {

	const char * kernelSource =
			"void kernel {{gHintCompiler}} softmax_forward(const __global float * restrict input , __constant int * labels , __global float * loss , __global int * nbRight, __global float *gradInput\n"
			"){\n"
					"  local float localTemp[{{batchSize}}];\n"
					"  local bool localBool[{{batchSize}}];\n"
					"  const int globalId = get_global_id(0);\n"
					"  int labelIdx = labels[globalId];\n"
					"  int imageOffset =globalId*{{numPlanes}};\n"
					"  float denominator=0;\n"
					"  float maxValue = input[imageOffset];\n"
					"\n"
					"  #pragma unroll\n"
					"  for(int plane = 1; plane < {{numPlanes}}; plane++) {\n"
			        "        maxValue = fmax(maxValue, input[imageOffset+plane]);\n"
			        "    }\n"
					"\n"
					"  #pragma unroll\n"
					"  for(int plane=0 ; plane<{{numPlanes}} ; plane++)\n"
					"    denominator+=exp(input[imageOffset+plane]-maxValue);\n"
					"\n"
					"  int selectorID= 0;\n"
					"  float maxValue2 = exp(input[imageOffset]-maxValue)/denominator;\n\n"
					"  gradInput[imageOffset]=maxValue2-(float)select(0,1,(labelIdx==0));\n\n"
					"  #pragma unroll\n"
					"  for(int plane=1;plane<{{numPlanes}};plane++)\n{"
					"    float temp=exp(input[imageOffset+plane]-maxValue)/denominator;\n"
					"    gradInput[imageOffset+plane]=temp-(float)select(0,1,(plane==labelIdx));\n"
					"    selectorID=select(selectorID,plane,(isgreater(temp,maxValue2)));\n"
					"    maxValue2=fmax(maxValue2,temp);\n"
					"  }\n"
					"  localTemp[globalId]=-log(exp(input[imageOffset+labelIdx]-maxValue)/denominator);\n\n"
					"  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n\n"
					"  if(globalId==0){\n"
					"    loss[0]=0;\n"
					"    for(int i=0;i<{{batchSize}};i++)\n"
					"      loss[0]+=localTemp[i];\n"
					"  }\n\n\n"
					"  localBool[globalId]=select(0,1,(labelIdx==selectorID));\n"
					"  barrier(CLK_LOCAL_MEM_FENCE);\n"
					"  if(globalId==0){\n"
					"    nbRight[0]=0;\n"
					"    for(int i=0;i<{{batchSize}};i++)\n"
					"      nbRight[0]+=select(0,1,localBool[i]);"
					"  }\n"
					"}\n"
					"\n"
					"";

    return kernelSource;
}

//////////////////////////////



void SoftMaxLayer::createKernelHalf(){

	buildKernelSoftmax_forwardHalf();

}

void SoftMaxLayer::buildKernelSoftmax_forwardHalf() {
    TemplatedKernel builder(ptrcl);


        setupBuilderSoftmax_forwardHalf(&builder);

        string identifier2="softmax_forwardHalf"+std::to_string(numPlanes);
        identifier2=identifier2+"_InputSize="+std::to_string(imageSize);
        identifier2=identifier2+"_batchsize="+std::to_string(batchSize);
        string test=getKernelTemplateSoftmax_forwardHalf();
        this->kernelHalf = builder.buildKernel(
           		identifier2,
               "softmax",
               test,
               "softmax_forward",
               false
        );
    }

void SoftMaxLayer::runSoftmax_forwardHalf(    CLWrapper *inputWrapper/*,CLWrapper *outputWrapper*/,int const *labels) {

#if TEST_SOFTMAX==1
clock_t startTimer1, stopTimer1,startTimer2, stopTimer2,startTimer3, stopTimer3;
	startTimer2=clock();
#endif

	//labelWrapper->copyToDevice(labels);

	labelWrapperHalf->copyToDevice_ZeroCopyObject_WriteFlag(labels);
	//->copyToDevice_ZeroCopyObject_WriteFlagHalf(labels);



#if TEST_SOFTMAX==1
	LOGI("globalSize %d workgroupsize %d",globalSize,workgroupsize);
	startTimer1=clock();
#endif

	kernelHalf->run_1d(globalSize, workgroupsize);
	//ptrcl->finish();
#if COMPAREHALF==1
	half *loss=lossWrapperHalf->map_ZeroCopyObject_ReadFlagHalf();
	int *nb=nbRightWrapperHalf->map_ZeroCopyObject_ReadFlagInt();
	half *grad=gradInputWrapperHalf->map_ZeroCopyObject_ReadFlagHalf();
	LOGI("loss %f nb right %d",HalfToFloat(loss[0]),nb[0]);
	for(int i=0;i<10;i++){
		LOGI("grad[%d]=%f",i,HalfToFloat(grad[i]));
	}
//	float e=0;
//	for(int i=0;i<previousLayer-> getOutputNumElements();i++){
//		e=abs(testgrad[i]-HalfToFloat(grad[i]));
//	}
//	LOGI("error %f",e);
	nbRightWrapperHalf->unMap_ZeroCopyObject_ReadFlagInt(nb);
	lossWrapperHalf->unMap_ZeroCopyObject_ReadFlagHalf(loss);
	gradInputWrapperHalf->unMap_ZeroCopyObject_ReadFlagHalf(grad);
#endif
	#if TEST_SOFTMAX==1
	stopTimer1 = clock();
	startTimer3=clock();
	#endif

	#if TEST_SOFTMAX==1

		stopTimer3 = clock();
		LOGI("read  took %g ms\n\n", 1000.0* (double)(stopTimer3 - startTimer3)/(double)CLOCKS_PER_SEC) ;
		LOGI("Softmax  took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		LOGI("loss %f", losshalf[0]) ;

		stopTimer2 = clock();
		LOGI("Softmax  took %g ms\n\n", 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC) ;
	#endif


}


void SoftMaxLayer::setupBuilderSoftmax_forwardHalf(TemplatedKernel *builder) {

	int possibleGlobalSize =batchSize;// batchSize;//crash because batchSize value = 0
	int possibleWorkgroupsize = std::min(possibleGlobalSize, ptrcl->getMaxWorkgroupSize());
	string hintCompilerString="__attribute__((vec_type_hint(";
	hintCompilerString+="half";
	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);


	builder->set("numPlanes",numPlanes);
	builder->set("batchSize",batchSize);

}

STATIC std::string SoftMaxLayer::getKernelTemplateSoftmax_forwardHalf() {

//	const char * kernelSource =
//			"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
//			"void kernel {{gHintCompiler}} softmax_forward(const __global half * restrict input , __constant int * labels , __global half * loss , __global int * nbRight, __global half *gradInput\n"
//			"){\n"
//			        "  const int globalId = get_global_id(0);\n"
//					"int imageOffset = globalId * {{numPlanes}};\n"
//				   "// first get the max\n"
//				   "half maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct\n"
//				   "for(int plane = 1; plane < {{numPlanes}}; plane++) {\n"
//					"   maxValue = fmax(maxValue, input[imageOffset + plane]);\n"
//				   "}\n"
//				   "// calculate sum, under this max\n"
//				   "half denominator = 0;\n"
//				   "for(int plane = 0; plane < {{numPlanes}}; plane++) {\n"
//					"   denominator += exp(input[imageOffset + plane] - maxValue);\n"
//				   "}\n"
//				   "// now calc the softmaxes:\n"
//					"int label = labels[globalId];\n"
//					"for(int plane = 0; plane < {{numPlanes}}; plane++) {\n"
//					"	gradInput[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;\n"
//					"}\n"
//					"gradInput[imageOffset + label] -= 1;\n"
//			"}";
//	const char * kernelSource =
//			"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
//			"void kernel {{gHintCompiler}} softmax_forward(const __global half * restrict input , __constant int * labels , __global half * loss , __global int * nbRight, __global half *gradInput\n"
//			"){\n"
//					"  local float localTemp[{{batchSize}}];\n"
//					"  local bool localBool[{{batchSize}}];\n"
//					"  const int globalId = get_global_id(0);\n"
//					"  int labelIdx = labels[globalId];\n"
//					"  int imageOffset =globalId*{{numPlanes}};\n"
//					"  float denominator=0;\n"
//					"  float maxValue = convert_float(input[imageOffset]);\n"
//					"\n"
//					"  #pragma unroll\n"
//					"  for(int plane = 1; plane < {{numPlanes}}; plane++) {\n"
//			        "        maxValue = fmax(maxValue, convert_float(input[imageOffset+plane]));\n"
//			        "    }\n"
//					"\n"
//					"  #pragma unroll\n"
//					"  for(int plane=0 ; plane<{{numPlanes}} ; plane++)\n"
//					"    denominator+=exp(convert_float(input[imageOffset+plane])-maxValue);\n"
//					"\n"
//					"  int selectorID= 0;\n"
//					"  float maxValue2 = exp(convert_float(input[imageOffset])-maxValue)/denominator;\n\n"
//					"  gradInput[imageOffset]=convert_half(maxValue2-(float)select(0,1,(labelIdx==0)));\n\n"
//					"  #pragma unroll\n"
//					"  for(int plane=1;plane<{{numPlanes}};plane++)\n{"
//					"    float temp=exp(convert_float(input[imageOffset+plane])-maxValue)/denominator;\n"
//					"    gradInput[imageOffset+plane]=convert_half(temp-(float)select(0,1,(plane==labelIdx)));\n"
//					"    selectorID=select(selectorID,plane,(isgreater(temp,maxValue2)));\n"
//					"    maxValue2=fmax(maxValue2,temp);\n"
//					"  }\n"
//					"  localTemp[globalId]=-log(exp(convert_float(input[imageOffset+labelIdx])-maxValue)/denominator);\n\n"
//					"  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n\n"
//					"  if(globalId==0){\n"
//			        "    float temp=0;\n"
//					"    for(int i=0;i<{{batchSize}};i++)\n"
//					"      temp+=localTemp[i];\n"
//					"    loss[0]=convert_float(temp);\n"
//					"  }\n\n\n"
//					"  localBool[globalId]=select(0,1,(labelIdx==selectorID));\n"
//					"  barrier(CLK_LOCAL_MEM_FENCE);\n"
//					"  if(globalId==0){\n"
//					"    nbRight[0]=0;\n"
//					"    for(int i=0;i<{{batchSize}};i++)\n"
//					"      nbRight[0]+=select(0,1,localBool[i]);"
//					"  }\n"
//					"}\n"
//					"\n"
//					"";


	const char * kernelSource =
			"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
			"void kernel {{gHintCompiler}} softmax_forward(const __global half * restrict input , __constant int * labels , __global half * loss , __global int * nbRight, __global half *gradInput\n"
			"){\n"
					"  local half localTemp[{{batchSize}}];\n"
					"  local bool localBool[{{batchSize}}];\n"
			        "  const int globalId = get_global_id(0);\n"
					"  int labelIdx = labels[globalId];\n"
					"  int imageOffset =globalId*{{numPlanes}};\n"
					"  half denominator=0;\n"
					"  half maxValue = input[imageOffset];\n"
					"\n"
					"  #pragma unroll\n"
					"  for(int plane = 1; plane < {{numPlanes}}; plane++) {\n"
					"        maxValue = fmax(maxValue, input[imageOffset+plane]);\n"
					"    }\n"
					"\n"
					"  #pragma unroll\n"
					"  for(int plane=0 ; plane<{{numPlanes}} ; plane++)\n"
		            "    denominator+=exp(input[imageOffset+plane]-maxValue);\n"
					"  int selectorID= 0;\n"
					"  half maxValue2 = exp(input[imageOffset]-maxValue)/denominator;\n\n"
					"  gradInput[imageOffset]=maxValue2-(half)select(0,1,(labelIdx==0));\n\n"
					"  #pragma unroll\n"
					"  for(int plane=1;plane<{{numPlanes}};plane++)\n{"
					"    half temp=exp(input[imageOffset+plane]-maxValue)/denominator;\n"
					"    gradInput[imageOffset+plane]=temp-(half)select(0,1,(plane==labelIdx));\n"
					"    selectorID=select(selectorID,plane,(isgreater(temp,maxValue2)));\n"
					"    maxValue2=fmax(maxValue2,temp);\n"
					"  }\n"
					"  localTemp[globalId]=-log(exp(input[imageOffset+labelIdx]-maxValue)/denominator);\n\n"
					"  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n\n"
					"  if(globalId==0){\n"
					"    loss[0]=0;\n"
					"    for(int i=0;i<{{batchSize}};i++)\n"
					"      loss[0]+=localTemp[i];\n"
			        "  }\n\n\n"
					"  localBool[globalId]=select(0,1,(labelIdx==selectorID));\n"
					"  barrier(CLK_LOCAL_MEM_FENCE);\n"
					"  if(globalId==0){\n"
					"    nbRight[0]=0;\n"
					"    for(int i=0;i<{{batchSize}};i++)\n"
					"      nbRight[0]+=select(0,1,localBool[i]);"
			        "  }\n"
					"}\n"
			        "\n"
			        "";

    return kernelSource;
}



