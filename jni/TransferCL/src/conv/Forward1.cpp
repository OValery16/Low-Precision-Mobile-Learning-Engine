// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "Forward1.h"
#include "../util/stringhelper.h"
#include "../../EasyCL/util/StatefulTimer.h"

#if TEST_FORWARD==1
#include "AddBias.h"
#endif


#include <sstream>
#include <iomanip>


using namespace std;



#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

static double TimeSpecToSeconds2(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}

VIRTUAL Forward1::~Forward1() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/Forward1.cpp: ~Forward1");
#endif
#if TEST_FORWARD==1
	delete kernelH;
	delete kernel;
    delete addBias;
#endif
#if EIGHT_BIT_ACCURACY ==0
	#if HALF_ACCURACY==0
		delete test;
	#else
		delete test2;
	#endif
#else
	delete test4;
#endif

}

#if TEST_FORWARD==1
std::string to_string_with_precision2(const float a_value, const int n = 2)
{
	std::stringstream ss;
	if (a_value==0)
		ss << std::fixed << 0;
	else
		ss << std::fixed << std::setprecision(n) << a_value;
    return ss.str();
}


float *Forward1::convolv(int batchSize, float *inputData, float *weights) {

	float *output = new float[ dim.outputCubeSize * batchSize ];
    for(int n = 0; n < batchSize; n++) {
        for(int filter = 0; filter < dim.numFilters; filter++) {
            for(int outRow = 0; outRow < dim.outputSize; outRow += 1 + dim.skip) {
                for(int outCol = 0; outCol < dim.outputSize; outCol += 1 + dim.skip) {
                    float sum = 0;
                    for(int inPlane = 0; inPlane < dim.inputPlanes; inPlane++) {
//                        cout << "inplane=" << inPlane << endl;
                        for(int u = -dim.halfFilterSize; u <= dim.halfFilterSize; u++) {
                            int inRow = outRow * (dim.skip + 1) + u + (dim.padZeros ? 0 : dim.halfFilterSize);
//                                cout << "candidate inRow " << inRow << endl;
                            if(inRow < 0 || inRow > dim.inputSize - 1) {
                                continue;
                            }
                            int filterRow = u + dim.halfFilterSize;
                            for(int v = -dim.halfFilterSize; v <= dim.halfFilterSize; v++) {
                                int inCol = outCol * (dim.skip + 1) + v + (dim.padZeros ? 0 : dim.halfFilterSize);
                                int filterCol = v + dim.halfFilterSize;
                                if(inCol < 0 || inCol > dim.inputSize - 1) {
                                    continue;
                                }
                                int inputIndex = (( n
                                    * dim.inputPlanes + inPlane)
                                    * dim.inputSize + inRow)
                                    * dim.inputSize + inCol;
                                int weightIndex = (( filter
                                    * dim.inputPlanes + inPlane)
                                    * dim.filterSize  + filterRow)
                                    * dim.filterSize  + filterCol;
//                                    cout << "inpos " << inRow << "," << inCol << " outpos " << outRow << "," << outCol
//                                        << " filterpos " << filterRow << "," << filterCol << endl;
                                float sumchange = inputData[ inputIndex] * weights[ weightIndex ];
                                if(sumchange != 0) {
//                                        cout << inputData[inputIndex] << " * " << weights[weightIndex] << " = " << sumchange << endl;
                                }
                                sum += sumchange;
//                                cout << "inputIndex=" << inputIndex << " weightIndex=" << weightIndex <<
//                                    "  inputData[inputIndex]=" << inputData[inputIndex] << " weights[weightIndex]=" << weights[weightIndex] << " sumchange " << sumchange << " sum=" << sum << endl;
                            }
                        }
                    }

//                    sum = fn->calc(sum);
                    int outputIndex = (( n
                        * dim.numFilters + filter)
                        * dim.outputSize + outRow)
                        * dim.outputSize + outCol;
                    output[outputIndex] = sum;
//                    cout << "outputIndex=" << outputIndex << " sum=" << sum << " output[outputIndex]=" <<
//                        output[outputIndex] << endl;
                }
            }
        }
    }
    return output;
}
#endif

VIRTUAL void Forward1::forwardFloat(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper){
#if TEST_FORWARD==1

//	float * output=(float*)outputWrapper->getHostArray();
	clock_t startTimer1, stopTimer1;
	startTimer1=clock();

	setup=false;

	if (setup!=true){
		globalSize = batchSize * dim.outputCubeSize;
		workgroupsize = kernel->get_kernel_work_group_size();
		//int workgroupsize = std::min(globalSize, cl->getMaxWorkgroupSize());
		globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
	}

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
    stopTimer1 = clock();
    LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

    startTimer1=clock();
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("Forward1::forward after call forward");

stopTimer1 = clock();

	LOGI("convolution took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

	startTimer1=clock();
    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("Forward1::forward END");
	stopTimer1 = clock();

	LOGI("bias took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

#endif
}

VIRTUAL int Forward1::getWorkgroupSize(int globalSize,int maxWorkgroupSizeSize){

	int workgroupSize=1;

	for(int i=maxWorkgroupSizeSize/2/*divided by 2 => ajout*/;i>0;i--){
		if (globalSize%i==0){
			workgroupSize=i;
			break;
		}
	}
	return workgroupSize;
}


VIRTUAL void Forward1::forwardHalf(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper){


#if TEST_FORWARD==1
	if (timeBenchmark)
		startTimer1=clock();
#endif

	float * selector =0;
	if (setup!=true){
		if (dim.useMaxPooling)
			globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
		else
			globalSize = batchSize * dim.outputCubeSize;


		workgroupsize = test->get_kernel_work_group_size();
		globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		setup = true;

		test->input(dataWrapper);
		test->input(weightsWrapper);
		test->output(outputWrapper);
		if(dim.biased)
			test->input(biasWrapper);

		if(normalization){
			LOGI("-------------------translate %f scale %f",dim.translate,dim.scale );
			float translate=dim.translate;
			float scale=dim.scale;
			test->input(translate);
			test->input(scale);

		}
		#if TRANSFER ==0
		if (dim.useMaxPooling){
			test->output(selectorWrapper);
			test->output(gradInputWrapper);
		}
		#endif
}



	#if TEST_FORWARD==1
		if (timeBenchmark){
			stopTimer1 = clock();
			LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		}


    if (timeBenchmark)
		startTimer1=clock();
    #endif
    test->run_1d(globalSize, workgroupsize);
    //cl->finish();

	#if TEST_FORWARD==1
    	cl->finish();
		if (timeBenchmark){
			StatefulTimer::timeCheck("Forward1::forward after call forward");
			stopTimer1 = clock();
			LOGI("convolution test took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

			stopTimer1 = clock();

		}
	#endif


}

VIRTUAL void Forward1::testHalf(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput){

#if TEST_FORWARD==1
	if (timeBenchmark)
		startTimer1=clock();
#endif
#if TRANSFER ==1
#if 0
	if ((dim.isLast==false)||(not cl->training)){
#endif
		float * selector =0;
		if (setupHalf!=true){
			if (dim.useMaxPooling)
				globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
			else
				globalSize = batchSize * dim.outputCubeSize;

			int testf = getWorkgroupSize(globalSize,test2->get_kernel_work_group_size());
			LOGI("test %d",testf);
			workgroupsize = testf;//test2->get_kernel_work_group_size();
			//globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
			setupHalf = true;


		test2->input(dataWrapper);
			test2->input(weightsWrapper);
			test2->output(outputWrapper);
			if(dim.biased)
				test2->input(biasWrapper);

			if(normalization){
				LOGI("-------------------half translate %f scale %f",dim.translate,dim.scale );
				half translate=FloatToHalf(dim.translate);
				half scale=FloatToHalf(dim.scale);
				test2->input(translate);
				test2->input(scale);

			}
			#if TRANSFER ==0
			if (dim.useMaxPooling){
				test2->output(selectorWrapper);
				test2->output(gradInputWrapper);
			}
			#endif
	}

//		half * temp7=biasWrapper->map_ZeroCopyObject_ReadFlagHalf();
//
//		for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+*/20/*)*/; i++){
//			LOGI("--------------biasWrapper[%d]=%f ",i,HalfToFloat(temp7[i]));
//		}
//		biasWrapper->unMap_ZeroCopyObject_ReadFlagHalf(temp7);

		#if TEST_FORWARD==1
			if (timeBenchmark){
				stopTimer1 = clock();
				LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
			}


		if (timeBenchmark)
			startTimer1=clock();
		#endif

		test2->run_1d(globalSize, workgroupsize);
//		cl->finish();
//		half * temp6=outputWrapper->map_ZeroCopyObject_ReadFlagHalf();
//
//		for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+*/20/*)*/; i++){
//			LOGI("outputWrapper[%d]=%f ",i,HalfToFloat(temp6[i]));
//		}
//		outputWrapper->unMap_ZeroCopyObject_ReadFlagHalf(temp6);

		#if TEST_FORWARD==1
			cl->finish();
			if (timeBenchmark){
				StatefulTimer::timeCheck("Forward1::forward after call forward");
				stopTimer1 = clock();
				LOGI("convolution test took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

				stopTimer1 = clock();

			}
		#endif
#if 0
	}else{
		float * selector =0;
		if (setupHalf!=true){
			getLastFCKernel(dataWrapper);
			if (dim.useMaxPooling)
				globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
			else
				globalSize = batchSize * dim.outputCubeSize;

			int testf = getWorkgroupSize(globalSize,test3->get_kernel_work_group_size());
			LOGI("test %d",testf);
			workgroupsize = testf;//test2->get_kernel_work_group_size();
			//globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
			setupHalf = true;
			LOGI("globalSize %d",globalSize);

		    test3->input(dataWrapper);
			test3->input(weightsWrapper);
			test3->output(outputWrapper);
			if(dim.biased)
				test3->input(biasWrapper);

			if(normalization){
				half translate=FloatToHalf(dim.translate);
				half scale=FloatToHalf(dim.scale);
				test3->input(translate);
				test3->input(scale);

			}
			if (dim.useMaxPooling){
				test3->output(selectorWrapper);
				test3->output(gradInputWrapper);
			}
	    }



		#if TEST_FORWARD==1
			if (timeBenchmark){
				stopTimer1 = clock();
				LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
			}


		if (timeBenchmark)
			startTimer1=clock();
		#endif

		struct timespec start1;
		struct timespec end1;
		struct timespec start2;
		struct timespec end2;
		double elapsedSeconds1;
		clock_gettime(CLOCK_MONOTONIC, &start1);

#if 0
		for(int i=0;i<128;i++)
		    test3->run_1d(globalSize, workgroupsize);
		cl->finish();
		clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds2(&end1) - TimeSpecToSeconds2(&start1);
		LOGI("3)time %f ms\n\n ",elapsedSeconds1*1000);
#else
		test3->run_1d(globalSize, workgroupsize);
		cl->finish();
		clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds2(&end1) - TimeSpecToSeconds2(&start1);
		LOGI("3)time %f ms\n\n ",elapsedSeconds1*1000);
#endif
//		half * p=dataWrapper->map_ZeroCopyObject_ReadFlagHalf();
//		for(int i=0;i<100;i++){
//				LOGI("Data[%d]=%f",i,HalfToFloat(p[i]));
//			}
////		for(int i=0;i<10;i++){
////				LOGI("A[%d]=%f",i,HalfToFloat(p[i+152]));
////			}
//		dataWrapper->unMap_ZeroCopyObject_ReadFlagHalf(p);
//
//
//		/*half **/ p=outputWrapper->map_ZeroCopyObject_ReadFlagHalf();
//		for(int i=0;i<100;i++){
//				LOGI("A[%d]=%f",i,HalfToFloat(p[i]));
//			}
////		for(int i=0;i<10;i++){
////				LOGI("A[%d]=%f",i,HalfToFloat(p[i+152]));
////			}
//		outputWrapper->unMap_ZeroCopyObject_ReadFlagHalf(p);


		#if TEST_FORWARD==1
			cl->finish();
			if (timeBenchmark){
				StatefulTimer::timeCheck("Forward1::forward after call forward");
				stopTimer1 = clock();
				LOGI("convolution test took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

				stopTimer1 = clock();

			}
		#endif
	}
#endif
#else
		float * selector =0;
		if (setupHalf!=true){
			if (dim.useMaxPooling)
				globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
			else
				globalSize = batchSize * dim.outputCubeSize;

			int testf = getWorkgroupSize(globalSize,test2->get_kernel_work_group_size());
			LOGI("test %d",testf);
			workgroupsize = testf;//test2->get_kernel_work_group_size();
			//globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
			setupHalf = true;


		test2->input(dataWrapper);
			test2->input(weightsWrapper);
			test2->output(outputWrapper);
			if(dim.biased)
				test2->input(biasWrapper);

			if(normalization){
				half translate=FloatToHalf(dim.translate);
				half scale=FloatToHalf(dim.scale);
				test2->input(translate);
				test2->input(scale);

			}
			if (dim.useMaxPooling){
				test2->output(selectorWrapper);
				test2->output(gradInputWrapper);
			}
	}



		#if TEST_FORWARD==1
			if (timeBenchmark){
				stopTimer1 = clock();
				LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
			}


	    if (timeBenchmark)
			startTimer1=clock();
	    #endif

	    test2->run_1d(globalSize, workgroupsize);
	    //cl->finish();

		#if TEST_FORWARD==1
	    	cl->finish();
			if (timeBenchmark){
				StatefulTimer::timeCheck("Forward1::forward after call forward");
				stopTimer1 = clock();
				LOGI("convolution test took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

				stopTimer1 = clock();

			}
		#endif
#endif

		/////////////////////////////////////////////

}



string generateString(half * data){


	int imMax=128;

	 string s2="";
	 for(int i=0;i<152;i=i+4){
		std::vector<string> myStrings;
		for(int im=0;im<imMax;im++){
			myStrings.push_back ("(half4)("+to_string(HalfToFloat(data[i+152*im]))+" , "+to_string(HalfToFloat(data[i+1+152*im]))+", "+to_string(HalfToFloat(data[i+2+152*im]))+" , "+to_string(HalfToFloat(data[i+3+152*im]))+")");
		}

		string s1="select("+myStrings[0]+","+myStrings[1]+",(short4)((exampleId==1),(exampleId==1),(exampleId==1),(exampleId==1)))";
		for(int i1=2;i1<imMax;i1++)
			s1= "select("+s1+","+myStrings[i1]+",(short4)((exampleId=="+to_string(i1)+"),(exampleId=="+to_string(i1)+"),(exampleId=="+to_string(i1)+"),(exampleId=="+to_string(i1)+")))";
		s1="half4 t"+to_string(i)+"="+ s1+";\n\n\n\n";

		s2=s2+s1;
	}

	std::ofstream out;
	string spath="/data/data/com.sony.openclexample1/directoryTest/testFC.txt";
    out.open(spath.c_str()/*"/data/data/com.sony.openclexample1/preloadingData/kernelcode.txt"*/, std::ios::app);
    out << s2;
	out.close();

    return s2;

}

void Forward1::getLastFCKernel(CLWrapper * dataWrapper) {
	struct timespec start1;
	struct timespec end1;
	struct timespec start2;
	struct timespec end2;
//	half *data0= dataWrapper->map_ZeroCopyObject_WriteFlagHalf();
//	for(int i=0;i<152;i++){
//			for(int im=0;im<128;im++)
//				data0[i+152*im]=FloatToHalf(im);
//			//LOGI("A[%d]=%f",i+152*im,HalfToFloat(data[i+152*im]));
//		}
//	dataWrapper->unMap_ZeroCopyObject_WriteFlagHalf(data0);

	half *data= dataWrapper->map_ZeroCopyObject_ReadFlagHalf();

	string s="";



		    const char * kernelSource =
				    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
		    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2HalfLastFC(\n"
		    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
		    	    "    global half *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
		    	    "    const int globalId = get_global_id(0);\n"
					"    int exampleId = ((globalId) ) / 10;\n"
					"    int filterId = ((globalId) ) % 10;\n"
					"    half4 sum = (half4)(convert_half(0.0f),convert_half(0.0f),convert_half(0.0f),convert_half(0.0f));\n"
					"                                  #pragma unroll\n"
					"    for (int planeId = 0; planeId < 38; planeId++) {\n"
					"            int inputPlaneIdx = exampleId * 38 + planeId * 1;\n"
					"            int filterIdx=filterId * 38 + planeId * 1;\n"
					"        sum += inputs[inputPlaneIdx] * filters[filterIdx] ;\n"
					"      }\n"
					"    output[globalId] = dot( sum,(half4)(convert_half(1.0f),convert_half(1.0f),convert_half(1.0f),convert_half(1.0f)))+bias[(globalId ) % 10 ];\n"
					"}\n";

//	const char * kernelSource =
//					    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
//			    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2HalfLastFC(\n"
//			    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
//			    	    "    global half *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
//			    	    "    const int globalId = get_global_id(0);\n"
//						"{{SelectoBegin}}\n"
//			    	    "\n"
//						"remplacer"
//			    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
//			    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
//			    	    "\n"
//						"    {{gVectorType}} sum = {{gVectorTypeInitialization}};\n"
//						"    {{gMaxPooling}}"
//			    	    "        {{gImageRowCoord}}"
//			    	    "        {{gImageColCoord}}"
//			    	    "        {{gBeginFirstLoop}}\n"
//			    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
//						"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
//			    		"        {{gInternalLoop}}"
//			    	    "      {{gEndFirstLoop}}\n"
//						"\n"
//						"    {{gMaxPoolingEnd}}\n"
//			    		"    output[globalId] = {{gresult}};\n"
//						"    {{outputPoolingSelector}}"
//			    	    "\n"
//						"{{SelectoEnd}}"
//			    	    "}\n"
//			    	    "\n"
//			    	    "";




	dataWrapper->unMap_ZeroCopyObject_ReadFlagHalf(data);
	string identifier3="convolve_imagecubes_float2HalfLastFC";
		identifier3=identifier3+"nbFilter=";
		identifier3=identifier3+std::to_string(dim.numFilters);
		identifier3=identifier3+"_InputSize="+std::to_string(dim.inputSize);
		identifier3=identifier3+"_batchsize="+std::to_string(dim.batchsize);
		identifier3=identifier3+"_OutputSize="+std::to_string(dim.outputSize);
		identifier3=identifier3+"_conv="+BoolToString(dim.isConv);
		identifier3=identifier3+"_normalize="+BoolToString(dim.needToNormalize);
		identifier3=identifier3+"_maxpool="+BoolToString(dim.useMaxPooling);





	  TemplatedKernel builderHalf(cl);

	  LOGI("getLastFCKernel");
	  setupBuilderConvolveHalf(&builderHalf, dim.batchsize);


	this->test3 = builderHalf.buildKernelOnline(
       		identifier3,
           "forward1.cl",
           kernelSource,
           "convolve_imagecubes_float2HalfLastFC",
           false, s
    );

//	test3=cl->buildKernelFromStringOnlineBuilding(kernelSource, "convolve_imagecubes_float2HalfLast", "", "forward1.cl", false);


}

#if 0
void Forward1::getLastFCKernel(CLWrapper * dataWrapper) {
	struct timespec start1;
	struct timespec end1;
	struct timespec start2;
	struct timespec end2;
//	half *data0= dataWrapper->map_ZeroCopyObject_WriteFlagHalf();
//	for(int i=0;i<152;i++){
//			for(int im=0;im<128;im++)
//				data0[i+152*im]=FloatToHalf(im);
//			//LOGI("A[%d]=%f",i+152*im,HalfToFloat(data[i+152*im]));
//		}
//	dataWrapper->unMap_ZeroCopyObject_WriteFlagHalf(data0);

	half *data= dataWrapper->map_ZeroCopyObject_ReadFlagHalf();
	for(int i=0;i<10;i++)
		LOGI("%f",HalfToFloat(data[i]));
	////////////////////
	char **aPtr;
	int len = dataWrapper->size(); // Start with 1 string
	aPtr = (char**)malloc(sizeof(char*) * len); // Do not cast malloc in C
	int *aPtrS=new int[len];
	double elapsedSeconds1;
	clock_gettime(CLOCK_MONOTONIC, &start1);
	int size=0;
	for(int i=0;i<dataWrapper->size();i++){
		aPtrS[i] = asprintf(&aPtr[i], "%3g", HalfToFloat(data[i]));
//		aPtr[i]=const_cast<char*>(to_string(i/*HalfToFloat(data[i])*/).c_str());
//		aPtrS[i]=to_string(HalfToFloat(data[i])).length();
	    size=size+aPtrS[i];
	}
	clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds2(&end1) - TimeSpecToSeconds2(&start1);
	LOGI("3)time %f\n\n size %d",elapsedSeconds1,size);
	///////////////////
	clock_gettime(CLOCK_MONOTONIC, &start2);
	table_str_t table(size);
	int i=0;
	for (size_t idx = 0; idx < /*dataWrapper->size()*/1; ++idx)
	{
		char c[1];
	  table.add("    half t", 10);
	  sprintf(c, "%d", idx);
	  table.add(c, 1);
	  table.add("=", 1);//convert_half(
	  table.add(aPtr[idx], aPtrS[idx]);
	  table.add(";\n", 3);
	  i=10+1+1+3+aPtrS[idx];
	}
	//table.add(")", 1);
	table.add("\0", 1);
	free(aPtr);
	delete[] aPtrS;
	clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds2(&end1) - TimeSpecToSeconds2(&start1);
	LOGI("3)time %f\n\n size %d",elapsedSeconds1,size);

	std::string str(table.m_str, table.m_str + i+1/*table.m_size*//*table.m_size*/);

	    char checkCharacter = ',';
	    int count = 0;

	    for (int i = 0; i < table.m_size; i++)
	    {
	        if (table.m_str[i] ==  checkCharacter)
	        {
	            ++ count;
	        }
	    }

	    LOGI("Number of %d",count);
	       ////////////////////////////////////


////////////////////////////////

//	    string s="\n";
//		for(int i=0;i<152;i=i+4){
//			s=s+"    half4 t"+to_string(i)+"=select((convert_half("+to_string(HalfToFloat(data[i]))+") , convert_half("+to_string(HalfToFloat(data[i+1]))+") , convert_half("+to_string(HalfToFloat(data[i+2]))+") , convert_half("+to_string(HalfToFloat(data[i+3]))+") ),(convert_half("+to_string(HalfToFloat(data[152+i]))+") , convert_half("+to_string(HalfToFloat(data[152+i+1]))+") , convert_half("+to_string(HalfToFloat(data[152+i+2]))+") , convert_half("+to_string(HalfToFloat(data[152+i+3]))+") ),exampleId==0);\n";
//		}
		string s=generateString(data);


	    //string s=str;//"    half t0=0.96582;\n";

	    LOGI("s [%s]",str.c_str());
	//string s="__constant half t["+to_string(10/*dataWrapper->size()*/)+"];\n"+str;
			//"half s["+to_string(dataWrapper->size())+"]={"+str+"};\n";

	//string s="half t[2]={5,5};\n";

//s="";



//	    const char * kernelSource =
//			    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
//	    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2HalfLastFC(\n"
//	    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
//	    	    "    global half *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
//	    	    "    const int globalId = get_global_id(0);\n"
//	    		"remplacer"
//	    "    int exampleId = ((globalId) ) / 10;\n"
//	    "    int filterId = ((globalId) ) % 10;\n"
//	    "    half4 sum = (half4)(convert_half(0.0f),convert_half(0.0f),convert_half(0.0f),convert_half(0.0f));\n"
//	    "                                  #pragma unroll\n"
//	    "    for (int planeId = 0; planeId < 38; planeId++) {\n"
//	    "            int inputPlaneIdx = exampleId * 38 + planeId * 1;\n"
//	    "            int filterIdx=filterId * 38 + planeId * 1;\n"
//	    "        sum += inputs[inputPlaneIdx] * filters[filterIdx] ;\n"
//	    "      }\n"
//	    "    output[globalId] = dot( sum,(half4)(convert_half(1.0f),convert_half(1.0f),convert_half(1.0f),convert_half(1.0f)))+bias[(globalId ) % 10 ];\n"
//	    "}\n";


	const char * kernelSource =
					    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
			    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2HalfLastFC(\n"
			    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
			    	    "    global half *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
			    	    "    const int globalId = get_global_id(0);\n"
						"{{SelectoBegin}}\n"
			    	    "\n"
			    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
			    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
			    	    "remplacer"
						"\n"
						"    {{gVectorType}} sum = {{gVectorTypeInitialization}};\n"
			    		"        {{gInternalLoop}}"
						"\n"
			    		"    output[globalId] = {{gresult}};\n"
			    	    "}\n"
			    	    "\n"
			    	    "";


	dataWrapper->unMap_ZeroCopyObject_ReadFlagHalf(data);
	string identifier3="convolve_imagecubes_float2HalfLastFC";
		identifier3=identifier3+"nbFilter=";
		identifier3=identifier3+std::to_string(dim.numFilters);
		identifier3=identifier3+"_InputSize="+std::to_string(dim.inputSize);
		identifier3=identifier3+"_batchsize="+std::to_string(dim.batchsize);
		identifier3=identifier3+"_OutputSize="+std::to_string(dim.outputSize);
		identifier3=identifier3+"_conv="+BoolToString(dim.isConv);
		identifier3=identifier3+"_normalize="+BoolToString(dim.needToNormalize);
		identifier3=identifier3+"_maxpool="+BoolToString(dim.useMaxPooling);





	  TemplatedKernel builderHalf(cl);

	  LOGI("getLastFCKernel");

	  //string internalLoopString2="sum += inputs[inputPlaneIdx] * filters[filterIdx] {{gCondition}};\n";
	  string internalLoopString2="sum=";
	  int i1=0;
	  for(int i=0;i<152-4;i=i+4){
		   //internalLoopString2+="t"+to_string(i)+"+";
		  internalLoopString2+="t"+to_string(i)+"*filters[filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} +"+to_string(i1)+"]+";
		  i1++;
	  }
	  i=152-4;
	  //internalLoopString2+="t"+to_string(i)+";";
	  internalLoopString2+="t"+to_string(i)+"*filters[filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} +"+to_string(i1)+"] {{gCondition}};";
	  setupBuilderConvolveHalf(&builderHalf, dim.batchsize);
	  builderHalf.set("gInternalLoop",internalLoopString2);

	this->test3 = builderHalf.buildKernelOnline(
       		identifier3,
           "forward1.cl",
           kernelSource,
           "convolve_imagecubes_float2HalfLastFC",
           false, s
    );

//	test3=cl->buildKernelFromStringOnlineBuilding(kernelSource, "convolve_imagecubes_float2HalfLast", "", "forward1.cl", false);


}
#endif


VIRTUAL void Forward1::testFloat(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput){


#if TEST_FORWARD==1
	if (timeBenchmark)
		startTimer1=clock();
#endif

	float * selector =0;
	if (setup!=true){
		if (dim.useMaxPooling)
			globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
		else
			globalSize = batchSize * dim.outputCubeSize;
		int testf = getWorkgroupSize(globalSize,test->get_kernel_work_group_size());
		LOGI("test float %d",testf);
		workgroupsize = testf;//test->get_kernel_work_group_size();
		//globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		setup = true;


		test->input(dataWrapper);
		test->input(weightsWrapper);
		test->output(outputWrapper);
		if(dim.biased)
			test->input(biasWrapper);

		if(normalization){
			LOGI("-------------------translate %f scale %f",dim.translate,dim.scale );
//			dim.translate=31.981195;
//			dim.scale=0.006146;
			float translate=dim.translate;
			float scale=dim.scale;
			test->input(translate);
			test->input(scale);

		}
		#if TRANSFER ==0
		if (dim.useMaxPooling){
			test->output(selectorWrapper);
			test->output(gradInputWrapper);
		}
		#endif
}



	#if TEST_FORWARD==1
		if (timeBenchmark){
			stopTimer1 = clock();
			LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		}


    if (timeBenchmark)
		startTimer1=clock();
    #endif

    test->run_1d(globalSize, workgroupsize);

#if 0
    cl->finish();
	float * temp6=outputWrapper->map_ZeroCopyObject_ReadFlag();

	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
		LOGI("outputWrapper[%d]=%f ",i,temp6[i]);
	}
	outputWrapper->unMap_ZeroCopyObject_ReadFlag(temp6);
#endif

	#if TEST_FORWARD==1
    	cl->finish();
		if (timeBenchmark){
			StatefulTimer::timeCheck("Forward1::forward after call forward");
			stopTimer1 = clock();
			LOGI("convolution test took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

			stopTimer1 = clock();

		}
	#endif


		/////////////////////////////////////////////

#if TEST_FORWARD==1
	if (timeBenchmark)
		startTimer1=clock();
#endif

}

void encode8BitWeight(float * array,int len, unsigned char *dst,float &minW,float &multW){

	    float min = 999.99, max = -999.99, mult;
		int i;
		for(i = 0; i < len; i++)
		{
			if(array[i] < min)
				min = array[i];
			if(array[i] > max){
				max = array[i];
				LOGI("max %f",max);
			}
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

void decode8BitWeight(float * array,int len, unsigned char *dst,float min,float mult){


	float invmult = mult ? 1.0 / mult : 0;
	for(int i = 0; i < len; i++)
		array[i] = dst[i] * invmult + min;

}



VIRTUAL void Forward1::forward8bitTestVersion(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput){

	half *temp6;
	half *temp7;
	float * temp8;
	float * temp9;
	float min;
	float mult;
	unsigned char* temp10;
#if HALF_ACCURACY==0
	///////////////////////////////

		//weight
		LOGI("--------------------weight");
		temp9=weightsWrapper->map_ZeroCopyObject_ReadFlag();
		unsigned char *dst = new unsigned char[dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize];
		float* array = new float[dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize];
		for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
			array[i]=temp9[i];
		}

		encode8BitWeight(array,dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize,dst, min,mult);
		CLWrapper* weightsWrapper8Bit= (CLWrapper*) cl->wrap(dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize, temp8);
		weightsWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
		temp8=weightsWrapper8Bit->map_ZeroCopyObject_WriteFlag();
		for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
			temp8[i]=(float)(dst[i]);
		}
		weightsWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp8);
//////////////
		CLWrapper* weightsWrapper8Bit2=(CLWrapper*) cl->wrap8Bit(dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize, temp10);
		weightsWrapper8Bit2->createZeroCopyObject_ReadWriteFlag_OnDevice();
		temp10=weightsWrapper8Bit2->map_ZeroCopyObject_WriteFlag8Bit();
		for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
			temp10[i]=dst[i];
		}
		weightsWrapper8Bit2->unMap_ZeroCopyObject_WriteFlag8Bit(temp10);

/////////////////////
		weightsWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
		////////////////////////////////////
	//data
		LOGI("--------------------data");
		float min2;
		float mult2;
		temp9=dataWrapper->map_ZeroCopyObject_ReadFlag();
	    unsigned char *dst2 = new unsigned char[batchSize * dim.inputCubeSize];
		float* array2 = new float[batchSize * dim.inputCubeSize];
		for(int i=0;i<batchSize * dim.inputCubeSize;i++){
			array2[i]=temp9[i];
		}
		encode8BitWeight(array2,batchSize * dim.inputCubeSize,dst2, min2,mult2);
		CLWrapper* dataWrapper8Bit= (CLWrapper*) cl->wrap(batchSize * dim.inputCubeSize, temp8);
		dataWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
		temp8=dataWrapper8Bit->map_ZeroCopyObject_WriteFlag();
		for(int i=0;i<batchSize * dim.inputCubeSize;i++){
			temp8[i]=(float)(dst2[i]);
		}
		dataWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp8);

		dataWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
		//////////////////
		CLWrapper* dataWrapper8Bit2=(CLWrapper*) cl->wrap8Bit(batchSize * dim.inputCubeSize, temp10);
		dataWrapper8Bit2->createZeroCopyObject_ReadWriteFlag_OnDevice();
		temp10=dataWrapper8Bit2->map_ZeroCopyObject_WriteFlag8Bit();
		for(int i=0;i<batchSize * dim.inputCubeSize;i++){
			temp10[i]=dst2[i];
		}
		dataWrapper8Bit2->unMap_ZeroCopyObject_WriteFlag8Bit(temp10);
///////////////////
		CLWrapper* outputWrapper8Bit= (CLWrapper*) cl->wrap(batchSize * dim.outputCubeSize, temp8);
		outputWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();

		CLWrapper* outputWrapper8Bit2=(CLWrapper*) cl->wrap8Bit(batchSize * dim.outputCubeSize, temp10);
		outputWrapper8Bit2->createZeroCopyObject_ReadWriteFlag_OnDevice();
		////////////////////////////////////
		CLWrapper* biasWrapper2= (CLWrapper*) cl->wrap(biasWrapper->size(), temp8);
		biasWrapper2->createZeroCopyObject_ReadWriteFlag_OnDevice();
		temp8=biasWrapper2->map_ZeroCopyObject_WriteFlag();
		temp9=biasWrapper->map_ZeroCopyObject_ReadFlag();
		for(int i=0;i<biasWrapper->size();i++){
			temp8[i]=temp9[i];
			}
		biasWrapper2->unMap_ZeroCopyObject_WriteFlag(temp8);
		biasWrapper->unMap_ZeroCopyObject_WriteFlag(temp9);

	////////////////////////////
#else
		///////////////////////////////

				//weight
				LOGI("--------------------weight");
				temp9=weightsWrapper->map_ZeroCopyObject_ReadFlag();
				unsigned char *dst = new unsigned char[dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize];
				float* array = new float[dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize];
				for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
					array[i]=HalfToFloat(temp9[i]);
				}

				encode8BitWeight(array,dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize,dst, min,mult);
				CLWrapper* weightsWrapper8Bit= (CLWrapper*) cl->wrap(dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize, temp8);
				weightsWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
				temp8=weightsWrapper8Bit->map_ZeroCopyObject_WriteFlag();
				for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
					temp8[i]=(float)(dst[i]);
				}
				weightsWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp8);
		//////////////
				CLWrapper* weightsWrapper8Bit2=(CLWrapper*) cl->wrap8Bit(dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize, temp10);
				weightsWrapper8Bit2->createZeroCopyObject_ReadWriteFlag_OnDevice();
				temp10=weightsWrapper8Bit2->map_ZeroCopyObject_WriteFlag8Bit();
				for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
					temp10[i]=dst[i];
				}
				weightsWrapper8Bit2->unMap_ZeroCopyObject_WriteFlag8Bit(temp10);

		/////////////////////
				weightsWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
				////////////////////////////////////
			//data
				LOGI("--------------------data");
				float min2;
				float mult2;
				temp9=dataWrapper->map_ZeroCopyObject_ReadFlag();
			    unsigned char *dst2 = new unsigned char[batchSize * dim.inputCubeSize];
				float* array2 = new float[batchSize * dim.inputCubeSize];
				for(int i=0;i<batchSize * dim.inputCubeSize;i++){
					array2[i]=HalfToFloat(temp9[i]);
				}
				encode8BitWeight(array2,batchSize * dim.inputCubeSize,dst2, min2,mult2);
				CLWrapper* dataWrapper8Bit= (CLWrapper*) cl->wrap(batchSize * dim.inputCubeSize, temp8);
				dataWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
				temp8=dataWrapper8Bit->map_ZeroCopyObject_WriteFlag();
				for(int i=0;i<batchSize * dim.inputCubeSize;i++){
					temp8[i]=(float)(dst2[i]);
				}
				dataWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp8);

				dataWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
				//////////////////
				CLWrapper* dataWrapper8Bit2=(CLWrapper*) cl->wrap8Bit(batchSize * dim.inputCubeSize, temp10);
				dataWrapper8Bit2->createZeroCopyObject_ReadWriteFlag_OnDevice();
				temp10=dataWrapper8Bit2->map_ZeroCopyObject_WriteFlag8Bit();
				for(int i=0;i<batchSize * dim.inputCubeSize;i++){
					temp10[i]=dst2[i];
				}
				dataWrapper8Bit2->unMap_ZeroCopyObject_WriteFlag8Bit(temp10);
		///////////////////
				CLWrapper* outputWrapper8Bit= (CLWrapper*) cl->wrap(batchSize * dim.outputCubeSize, temp8);
				outputWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();

				CLWrapper* outputWrapper8Bit2=(CLWrapper*) cl->wrap8Bit(batchSize * dim.outputCubeSize, temp10);
				outputWrapper8Bit2->createZeroCopyObject_ReadWriteFlag_OnDevice();
				////////////////////////////////////
				CLWrapper* biasWrapper2= (CLWrapper*) cl->wrap(biasWrapper->size(), temp8);
				biasWrapper2->createZeroCopyObject_ReadWriteFlag_OnDevice();
				temp8=biasWrapper2->map_ZeroCopyObject_WriteFlag();
				temp9=biasWrapper->map_ZeroCopyObject_ReadFlag();
				for(int i=0;i<biasWrapper->size();i++){
					temp8[i]=HalfToFloat(temp9[i]);
					}
				biasWrapper2->unMap_ZeroCopyObject_WriteFlag(temp8);
				biasWrapper->unMap_ZeroCopyObject_WriteFlag(temp9);

			////////////////////////////


/////////////////////////////////
//
//	//weight
//	temp6=weightsWrapper->map_ZeroCopyObject_ReadFlagHalf();
//
//	unsigned char *dst = new unsigned char[weightsWrapper->size()];
//	float* array = new float[weightsWrapper->size()];
//
//	for(int i=0;i<weightsWrapper->size();i++){
//		array[i]=HalfToFloat(temp6[i]);
//	}
//	encode8BitWeight(array,weightsWrapper->size(),dst, min,mult);
//
//	CLWrapper* weightsWrapper8Bit= (CLWrapper*) cl->wrap(dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize, temp8);
//	weightsWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
//	temp8=weightsWrapper8Bit->map_ZeroCopyObject_WriteFlag();
//	for(int i=0;i<weightsWrapper->size();i++){
//		temp8[i]=(float)(dst[i]);
//	}
//	weightsWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp8);
//
//	weightsWrapper->unMap_ZeroCopyObject_ReadFlagHalf(temp6);
//
//	////////////////////////////////////
////data
//	float min2;
//	float mult2;
//	temp6=dataWrapper->map_ZeroCopyObject_ReadFlagHalf();
//
//    unsigned char *dst2 = new unsigned char[batchSize * dim.inputCubeSize];
//	float* array2 = new float[batchSize * dim.inputCubeSize];
//
//	for(int i=0;i<batchSize * dim.inputCubeSize;i++){
//		array2[i]=HalfToFloat(temp6[i]);
//	}
//	encode8BitWeight(array2,batchSize * dim.inputCubeSize,dst2, min2,mult2);
//
//	CLWrapper* dataWrapper8Bit= (CLWrapper*) cl->wrap(batchSize * dim.inputCubeSize, temp8);
//	dataWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
//	temp8=dataWrapper8Bit->map_ZeroCopyObject_WriteFlag();
//	for(int i=0;i<batchSize * dim.inputCubeSize;i++){
//		temp8[i]=(float)(dst2[i]);
//	}
//	dataWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp8);
//
//
//	dataWrapper->unMap_ZeroCopyObject_ReadFlagHalf(temp6);
//
//	CLWrapper* outputWrapper8Bit= (CLWrapper*) cl->wrap(outputWrapper->size(), temp8);
//	outputWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
//	////////////////////////////////////
//	CLWrapper* biasWrapper2= (CLWrapper*) cl->wrap(biasWrapper->size(), temp8);
//	biasWrapper2->createZeroCopyObject_ReadWriteFlag_OnDevice();
//	temp8=biasWrapper2->map_ZeroCopyObject_WriteFlag();
//	temp6=biasWrapper->map_ZeroCopyObject_ReadFlagHalf();
//	for(int i=0;i<biasWrapper->size();i++){
//		temp8[i]=HalfToFloat(temp6[i]);
//		}
//
//	biasWrapper2->unMap_ZeroCopyObject_WriteFlag(temp8);
//	biasWrapper->unMap_ZeroCopyObject_WriteFlag(temp8);
//////////////////////////////
#endif
	temp10=weightsWrapper8Bit2->map_ZeroCopyObject_ReadFlag8Bit();
	for(int i=0; i<20;i++){
			LOGI("Weight[%d]=%d data2[%d]=%d",i,(int)(temp10[i]),i,(int)(dst[i]));
		}
	weightsWrapper8Bit2->unMap_ZeroCopyObject_ReadFlag8Bit(temp10);
	LOGI("-----------------------------");
	temp10=dataWrapper8Bit2->map_ZeroCopyObject_ReadFlag8Bit();
	for(int i=0; i<20;i++){
			LOGI("data1[%d]=%d data2[%d]=%d",i,(int)(temp10[i]),i,(int)(dst2[i]));
		}
	dataWrapper8Bit2->unMap_ZeroCopyObject_ReadFlag8Bit(temp10);
	float multR;
	float minR;
	float multW;
	float minW;
	float multI;
	float minI;
	ExtractMinMultFromFile(minW, multW,minI, multI,minR, multR);
	LOGI("minW=%f, multW=%f,minI=%f, multI=%f,minR=%f, multR=%f",minW, multW,minI, multI,minR, multR);

		////////
	float * selector =0;
	if (setupEightBit!=true){
		if (dim.useMaxPooling)
			globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
		else
			globalSize = batchSize * dim.outputCubeSize;
		int testf = getWorkgroupSize(globalSize,test->get_kernel_work_group_size());

		workgroupsize = testf;//test->get_kernel_work_group_size();
		//globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		setupEightBit = true;



		/////////////////////////////////////
		float invmult2 = mult2 ? 1.0 / mult2 : 0;
		float invmult = mult ? 1.0 / mult : 0;
		float invmult3 = 1.0 / (mult*mult2);
		LOGI("-----------invmult=%f",invmult);
		LOGI("-----------invmult2=%f",invmult2);
		LOGI("-----------invmult2=%f",invmult3);
//		test4->input(min2);
//		test4->input(min);
//		test4->input(invmult2);
//		test4->input(invmult);
		test4->input(dataWrapper8Bit2);
		test4->input(weightsWrapper8Bit2);
		test4->output(outputWrapper8Bit2);
		if(dim.biased)
			test4->input(biasWrapper2/*biasWrapper*/);
		if(normalization){
//			half translate=FloatToHalf(dim.translate);
//			half scale=FloatToHalf(dim.scale);
			test4->input(dim.translate);
			test4->input(dim.scale);
		}
		if (dim.useMaxPooling){
			test4->output(selectorWrapper);
			test4->output(gradInputWrapper);
		}
	}
	LOGI("5");
	test4->run_1d(globalSize, workgroupsize);
	cl->finish();
	LOGI("6");
	temp10=outputWrapper8Bit2->map_ZeroCopyObject_ReadFlag8Bit();
LOGI("7");
	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
		LOGI("RAW output8Bit[%d]=%f ",i,((float)temp10[i]));
	}
	outputWrapper8Bit2->unMap_ZeroCopyObject_ReadFlag8Bit(temp10);



	temp10=outputWrapper8Bit2->map_ZeroCopyObject_ReadFlag8Bit();
LOGI("7");
	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
		LOGI("output8Bit[%d]=%f ",i,((float)temp10[i])/multR+minR);
	}
	outputWrapper8Bit2->unMap_ZeroCopyObject_ReadFlag8Bit(temp10);



#if 1
	temp8=dataWrapper8Bit->map_ZeroCopyObject_ReadFlag();
	float invmult = mult2 ? 1.0 / mult2 : 0;

	LOGI("----------COMPARE---------------\n");

	for(int i=0; i<10;i++){
		LOGI("data1[%d]=%f data2[%d]=%f",i,array2[i],i,temp8[i]* invmult + min2);
	}
	dataWrapper8Bit->unMap_ZeroCopyObject_ReadFlag(temp8);
#endif
	delete weightsWrapper8Bit2;
	delete dataWrapper8Bit;
	delete outputWrapper8Bit;
	delete weightsWrapper8Bit;
	delete biasWrapper2;
	delete[] array;
	delete[] dst;
	delete[] array2;
	delete[] dst2;
}

VIRTUAL void Forward1::ExtractMinMultFromData(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper * tempOutput){

	half *temp6;
	half *temp7;
	float * temp8;
	float * temp9;
	float min;
	float mult;
	unsigned char* temp10;

	//weight
	temp9=weightsWrapper->map_ZeroCopyObject_ReadFlag();
	unsigned char *dst = new unsigned char[dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize];
	float* array = new float[dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize];
	for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
		array[i]=temp9[i];
	}

	encode8BitWeight(array,dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize,dst, min,mult);


	weightsWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
	////////////////////////////////////
//data

	float min2;
	float mult2;
	temp9=dataWrapper->map_ZeroCopyObject_ReadFlag();
	unsigned char *dst2 = new unsigned char[batchSize *dim.inputCubeSize];
	float* array2 = new float[batchSize *dim.inputCubeSize];
	for(int i=0;i<batchSize *dim.inputCubeSize;i++){
		array2[i]=temp9[i];

	}
	encode8BitWeight(array2,batchSize * dim.inputCubeSize,dst2, min2,mult2);


	dataWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);


	string s ="Conv "+to_string(dim.layerId)+" : Weight (min:"+to_string(min)+",mult:"+to_string(mult)+"),Data (min:"+to_string(min2)+",mult:"+to_string(mult2)+")\n";
	string s2=cl->absolutePath+"minMult2.txt";
	ofstream myfile;
	myfile.open (s2,ios::out | ios::app);
	myfile << s;
	myfile.close();


	delete[] array;
	delete[] dst;
	delete[] array2;
	delete[] dst2;
}

VIRTUAL void Forward1::ExtractMinMultFromFile(float &minW, float &multW,float &minI, float &multI,float &minR, float &multR){
#if 0
	//////////
	string s ="Conv "+to_string(dim.layerId)+" : Weight (min:"+to_string(min)+",mult:"+to_string(mult)+"),Data (min:"+to_string(min2)+",mult:"+to_string(mult2)+")\n";
	string s2=cl->absolutePath+"minMult.txt";
	ofstream myfile;
	myfile.open (s2,ios::out | ios::app);
	myfile << s;
	myfile.close();
#else
	multR=1;
	minR=0;
	multW=1;
	minW=0;
	multI=1;
	minI=0;
	string s2=cl->absolutePath+"minMult.txt";
		string s3=" ";
		string s4="Conv "+to_string(dim.layerId);
		string line;
		  ifstream myfile (s2);
		  if (myfile.is_open())
		  {
		      while ( getline (myfile,line) )
		      {
		    	  if  (line.find(s4)!=std::string::npos){
		    		  s3=line;
		    	  }

		      }
		      myfile.close();
		  }

		  s4=s3.substr(s3.find("Weight (min:"));

		  minW=std::stod(s4.substr(12,s3.find(",")));


	  multW=std::stod(s4.substr(s4.find(",")+6,s4.find(")")-s4.find(",")-1).c_str());

	/////////////////////////////////
	s2=cl->absolutePath+"minMult.txt";
	s3=" ";
	s4="Conv "+to_string(dim.layerId);
	ifstream myfile2 (s2);
	if (myfile2.is_open()){
	  while ( getline (myfile2,line) ){
		  if  (line.find(s4)!=std::string::npos){
			  s3=line;
		  }
	  }
	  myfile2.close();
	}

	s4=s3.substr(s3.find("Data (min:"));

	minI=std::stod(s4.substr(10,s3.find(",")));


	multI=std::stod(s4.substr(s4.find(",")+6,s4.find(")")-s4.find(",")-1).c_str());
	/////////////////////////////////

	if (not dim.isLast){
		s2=cl->absolutePath+"minMult.txt";
		s3=" ";
		s4="Conv "+to_string(dim.layerId);
		ifstream myfile3 (s2);
		  if (myfile3.is_open())
		  {
			  while ( getline (myfile3,line) )
			  {
	//		    	  LOGI("--------------find [%s] display [%s] %d",s4.c_str(),line.c_str(),line.find(s4));

				  if  (line.find(s4)!=std::string::npos){
					  getline (myfile3,line);
					  s3=line;
				  }

			  }
			  myfile3.close();
		  }

		  s4=s3.substr(s3.find("Data (min:"));

		  minR=std::stod(s4.substr(10,s3.find(",")));

	  multR=std::stod(s4.substr(s4.find(",")+6,s4.find(")")-s4.find(",")-1).c_str());
	}
#endif
}


VIRTUAL void Forward1::forward8bit(int batchSize, CLWrapper *dataWrapper8Bit, CLWrapper *weightsWrapper8Bit, CLWrapper *biasWrapper,CLWrapper *outputWrapper8Bit, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper){
//LOGI("forward8bit");

#if 0
	half * temp7=biasWrapper->map_ZeroCopyObject_ReadFlagHalf();

	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+*/20/*)*/; i++){
		LOGI("++++++++++biasWrapper[%d]=%f ",i,HalfToFloat(temp7[i]));
	}
	biasWrapper->unMap_ZeroCopyObject_ReadFlagHalf(temp7);

	unsigned char* temp1;
	temp1=dataWrapper8Bit->map_ZeroCopyObject_ReadFlag8Bit();
	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
		LOGI("RAW input8Bit[%d]=%f ",i,((float)temp1[i]/multI+minI));
	}
	dataWrapper8Bit->unMap_ZeroCopyObject_ReadFlag8Bit(temp1);

	temp1=weightsWrapper8Bit->map_ZeroCopyObject_ReadFlag8Bit();
	for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
		LOGI("RAW weight8Bit[%d]=%f ",i,((float)temp1[i]/multW+minW));
	}
	weightsWrapper8Bit->unMap_ZeroCopyObject_ReadFlag8Bit(temp1);
#endif
	float * selector =0;
	if (setupEightBit!=true){
		if (dim.useMaxPooling)
			globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
		else
			globalSize = batchSize * dim.outputCubeSize;
		int testf = getWorkgroupSize(globalSize,test4->get_kernel_work_group_size());

		workgroupsize = testf;//test->get_kernel_work_group_size();
		//globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		setupEightBit = true;


//		LOGI("1");
		test4->input(dataWrapper8Bit);LOGI("1");
		test4->input(weightsWrapper8Bit);LOGI("2");
		test4->output(outputWrapper8Bit);LOGI("3");
		if(dim.biased)
			test4->input(biasWrapper);LOGI("4");
		if(normalization){
//			LOGI("translate %f scale %f",dim.translate,dim.scale );
//						dim.translate=31.981195;
//						dim.scale=0.006146;
			test4->input(dim.translate);LOGI("5");
			test4->input(dim.scale);LOGI("6");
		}
		#if TRANSFER ==0
		if (dim.useMaxPooling){
			test4->output(selectorWrapper);LOGI("7");
			test4->output(gradInputWrapper);LOGI("8");
		}
		#endif
	}

	test4->run_1d(globalSize, workgroupsize);
	//cl->finish();
#if 0
	if (not dim.isLast){
		unsigned char* temp;
		temp=outputWrapper8Bit->map_ZeroCopyObject_ReadFlag8Bit();
		for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
			LOGI("RAW output8Bit[%d]=%f ",i,((float)temp[i]/multR+minR));
		}
		outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlag8Bit(temp);
	}else{
		#if HALF_ACCURACY==0
			float* temp;
			temp=outputWrapper8Bit->map_ZeroCopyObject_ReadFlag();
			for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
				LOGI("RAW output8Bit[%d]=%f ",i,temp[i]);
			}
			outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlag(temp);
		#else
			half* temp;
			temp=outputWrapper8Bit->map_ZeroCopyObject_ReadFlagHalf();
			for(int i=0/*dim.outputCubeSize*/; i</*(dim.outputCubeSize+20)*/20; i++){
				LOGI("---------half RAW output8Bit[%d]=%f ",i,HalfToFloat(temp[i]));
			}
			outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlagHalf(temp);
		#endif
	}

#endif


}


VIRTUAL void Forward1::forwardChooseMethod(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper, CLWrapper * tempInput, CLWrapper *weightsWrapper2, CLWrapper *biasWrapper2, CLWrapper * tempOutput, CLWrapper *gradInputWrapperHalf,CLWrapper *dataWrapper8Bit, CLWrapper *weightsWrapper8Bit, CLWrapper *biasWrapper8Bit,
    CLWrapper *outputWrapper8Bit){

////
////
	//ExtractMinMultFromData( 1, dataWrapper, weightsWrapper,biasWrapper,outputWrapper, selectorWrapper,gradInputWrapper, tempInput, tempOutput);

	half *temp6;
	float*tempOriginal;
	half * test3=0;
	#if EIGHT_BIT_ACCURACY==1
		#if HALF_ACCURACY==0
			if (not cl->fit8BitData){
				//if (not dim.isLast){
						forward8bit(batchSize, dataWrapper8Bit, weightsWrapper8Bit, biasWrapper8Bit,outputWrapper8Bit,selectorWrapper, gradInputWrapper);
//				}else{
//					float * temp10;
//					float * temp9;
//					temp9=weightsWrapper->map_ZeroCopyObject_ReadFlag();
//					temp10=weightsWrapper8Bit->map_ZeroCopyObject_WriteFlag();
//					for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
//						temp10[i]=temp9[i];
//					}
//					weightsWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp10);
//					weightsWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
//					////////
//					temp9=biasWrapper->map_ZeroCopyObject_ReadFlag();
//					temp10=biasWrapper8Bit->map_ZeroCopyObject_WriteFlag();
//					for(int i=0;i<dim.numFilters;i++){
//						temp10[i]=temp9[i];
//					}
//					biasWrapper8Bit->unMap_ZeroCopyObject_WriteFlag(temp10);
//					biasWrapper->unMap_ZeroCopyObject_ReadFlag(temp9);
//					/////////////////
//					float* temp8;
//					//CLWrapper* outputWrapper8Bit= (CLWrapper*) cl->wrap(batchSize * dim.outputCubeSize, temp8);
//					//outputWrapper8Bit->createZeroCopyObject_ReadWriteFlag_OnDevice();
//
//					forward8bit(batchSize, dataWrapper8Bit, weightsWrapper8Bit, biasWrapper8Bit,outputWrapper8Bit/*tempOutput*/,selectorWrapper, gradInputWrapperHalf);
//					//delete outputWrapper8Bit;
//				}

		#else
			if (not cl->fit8BitData){
				//if (not dim.isLast){
					forward8bit(batchSize, dataWrapper8Bit, weightsWrapper8Bit, biasWrapper8Bit,outputWrapper8Bit,selectorWrapper, gradInputWrapperHalf);
//				}else{
//						half * temp10;
//						half * temp9;
//						temp9=weightsWrapper2->map_ZeroCopyObject_ReadFlagHalf();
//						temp10=weightsWrapper8Bit->map_ZeroCopyObject_WriteFlagHalf();
//						for(int i=0;i<dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;i++){
//							temp10[i]=temp9[i];
//						}
//						weightsWrapper8Bit->unMap_ZeroCopyObject_WriteFlagHalf(temp10);
//						weightsWrapper2->unMap_ZeroCopyObject_ReadFlagHalf(temp9);
//						////////////////////
//						//copy weight
//						//it is for test
//						half * temp11;
//						half * temp12;
//						temp11=biasWrapper2->map_ZeroCopyObject_ReadFlagHalf();
//						temp12=biasWrapper8Bit->map_ZeroCopyObject_WriteFlagHalf();
//						for(int i=0;i<dim.numFilters;i++){
//							temp12[i]=temp11[i];
//						}
//						biasWrapper8Bit->unMap_ZeroCopyObject_WriteFlagHalf(temp12);
//						biasWrapper2->unMap_ZeroCopyObject_ReadFlagHalf(temp11);
//						/////////////////
//						half* temp8;
//						forward8bit(batchSize, dataWrapper8Bit, weightsWrapper8Bit, biasWrapper8Bit,outputWrapper8Bit/*tempOutput*/,selectorWrapper, gradInputWrapperHalf);
//				}

		#endif
		}else{
			#if HALF_ACCURACY==0
				ExtractMinMultFromData(batchSize, dataWrapper, /*weightsWrapperHalf2*/weightsWrapper, /*biasWrapperHalf2*/biasWrapper,tempOutput, selectorWrapper, gradInputWrapper, tempInput, tempOutput);
			#endif
		}
	#else


	#if COMPARE_VALUE==1
		temp6=tempInput->map_ZeroCopyObject_ReadFlagHalf();
		tempOriginal=dataWrapper->map_ZeroCopyObject_ReadFlag();
		for(int i =0; i<10;i++){
			LOGI("%f %f",tempOriginal[i],HalfToFloat(temp6[i]));
		}
		dataWrapper->unMap_ZeroCopyObject_ReadFlag(tempOriginal);
		tempInput->unMap_ZeroCopyObject_ReadFlagHalf(temp6);

		LOGI("//////////////////////////////////");

		temp6=weightsWrapper2->map_ZeroCopyObject_ReadFlagHalf();
		tempOriginal=weightsWrapper->map_ZeroCopyObject_ReadFlag();
		for(int i =0; i<10;i++){
			LOGI("%f %f",tempOriginal[i],HalfToFloat(temp6[i]));
		}
		weightsWrapper->unMap_ZeroCopyObject_ReadFlag(tempOriginal);
		weightsWrapper2->unMap_ZeroCopyObject_ReadFlagHalf(temp6);
	#endif
	#if TRANSFER ==1
		#if HALF_ACCURACY==0
			testFloat( batchSize, dataWrapper, weightsWrapper, biasWrapper,
					outputWrapper, selectorWrapper, gradInputWrapper, tempInput, tempOutput);
		#else
			//if (cl->training){
				testHalf(batchSize, tempInput, /*weightsWrapperHalf2*/weightsWrapper2, /*biasWrapperHalf2*/biasWrapper2,
						tempOutput, selectorWrapper, gradInputWrapperHalf, tempInput, tempOutput);
				//cl->finish();
	//		}else{
	//			testFloat( batchSize, dataWrapper, weightsWrapper, biasWrapper,
	//				outputWrapper, selectorWrapper, gradInputWrapper, tempInput, tempOutput);
	//		}
		#endif
	#else
		#if HALF_ACCURACY==0
			testFloat( batchSize, dataWrapper, weightsWrapper, biasWrapper,
					outputWrapper, selectorWrapper, gradInputWrapper, tempInput, tempOutput);
		#else
				testHalf(batchSize, tempInput, /*weightsWrapperHalf2*/weightsWrapper2, /*biasWrapperHalf2*/biasWrapper2,
						tempOutput, selectorWrapper, gradInputWrapperHalf, tempInput, tempOutput);
		#endif
	#endif
#endif
/////////////////
#if COMPARE_VALUE==1
//	delete biasWrapperHalf2;
//	delete weightsWrapperHalf2;

temp6=tempOutput->map_ZeroCopyObject_ReadFlagHalf();
tempOriginal=outputWrapper->map_ZeroCopyObject_ReadFlag();
for(int i=0;i<10 ;i++){
	LOGI("-----------------------[%f vs %f",tempOriginal[i],HalfToFloat(temp6[i]));
}
cl->finish();
float error=0;
	float sum=0;
	float sum2=0;
//if (globalSize>batchSize * dim.outputCubeSize){
//	LOGI("1");
////for(int i =0; i<10;i++){
////	LOGI("%f %f",tempOriginal[i],HalfToFloat(temp6[i]));
////}
//	for(int i=0;i<batchSize * dim.outputCubeSize ;i++){
//	error=error+abs(tempOriginal[i]-HalfToFloat(temp6[i]));
//	}
//	for(int i=0;i<batchSize * dim.outputCubeSize ;i++){
//	sum=sum+tempOriginal[i];
//	}
//	for(int i=0;i<batchSize * dim.outputCubeSize ;i++){
//	sum2=sum2+HalfToFloat(temp6[i]);
//	}
//}else{
//	LOGI("2");
	for(int i=0;i<globalSize ;i++){
	error=error+abs(tempOriginal[i]-HalfToFloat(temp6[i]));
	}
	for(int i=0;i<globalSize ;i++){
	sum=sum+tempOriginal[i];
	}
	for(int i=0;i<globalSize ;i++){
	sum2=sum2+HalfToFloat(temp6[i]);
	}
//}


LOGI("sum %f vs %f size %d %d %d error %f",sum,sum2,batchSize * dim.outputCubeSize,globalSize,workgroupsize,error);
LOGI("------------------");
////if(error>1000){
//for(int i=0;i<20 ;i++){
//	LOGI("-----------------------[%f vs %f",tempOriginal[i],HalfToFloat(temp6[i]));
//}
//}
outputWrapper->unMap_ZeroCopyObject_ReadFlag(tempOriginal);
tempOutput->unMap_ZeroCopyObject_ReadFlagHalf(temp6);
cl->finish();
#endif

#if 0//COMPARE_EIGHT_BIT_ACCURACY==1



#if HALF_ACCURACY==0
float error=0;
	float sum=0;
	float sum2=0;

	if (not dim.isLast){
		unsigned char* temp8;

		temp8=outputWrapper8Bit->map_ZeroCopyObject_ReadFlag8Bit();
		tempOriginal=outputWrapper->map_ZeroCopyObject_ReadFlag();
		for(int i=0;i<10 ;i++){
			LOGI("-----------------------[%f vs %f",tempOriginal[i],((float)temp8[i]/multR+minR));
		}

		for(int i=0;i<globalSize ;i++){
		error=error+abs(tempOriginal[i]-((float)temp8[i]/multR+minR));
		}
		for(int i=0;i<globalSize ;i++){
		sum=sum+tempOriginal[i];
		}
		for(int i=0;i<globalSize ;i++){
			sum2=sum2+((float)temp8[i]/multR+minR);
		}
		outputWrapper->unMap_ZeroCopyObject_ReadFlag(tempOriginal);
		outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlag8Bit(temp8);
	}else{
		float* temp8;
		temp8=outputWrapper8Bit->map_ZeroCopyObject_ReadFlag();
		tempOriginal=outputWrapper->map_ZeroCopyObject_ReadFlag();
		LOGI("globalSize=%d %d",globalSize,batchSize * dim.outputCubeSize);
		for(int i=0;i<10 ;i++){
			LOGI("-----------------------[%f vs %f",tempOriginal[i],temp8[i]);
		}
		for(int i=0;i<globalSize ;i++){

			error=error+abs(tempOriginal[i]-temp8[i]);
		}
		for(int i=0;i<globalSize ;i++){
		sum=sum+tempOriginal[i];
		}
		for(int i=0;i<globalSize ;i++){
		sum2=sum2+temp8[i];
		}
		outputWrapper->unMap_ZeroCopyObject_ReadFlag(tempOriginal);
		outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlag(temp8);
	}


LOGI("------------------sum %f vs %f size %d %d %d error %f",sum,sum2,batchSize * dim.outputCubeSize,globalSize,workgroupsize,error/globalSize);
LOGI("------------------");
#else
float error=0;
	float sum=0;
	float sum2=0;
	LOGI("----end=======");
	if (not dim.isLast){
		unsigned char* temp8;

		temp8=outputWrapper8Bit->map_ZeroCopyObject_ReadFlag8Bit();
		temp6=tempOutput->map_ZeroCopyObject_ReadFlagHalf();
		for(int i=0;i<10 ;i++){
			LOGI("-----------------------[%f vs %f",HalfToFloat(temp6[i]),((float)temp8[i]/multR+minR));
		}

		for(int i=0;i<globalSize ;i++){
		error=error+abs(HalfToFloat(temp6[i])-((float)temp8[i]/multR+minR));
		}
		for(int i=0;i<globalSize ;i++){
		sum=sum+HalfToFloat(temp6[i]);
		}
		for(int i=0;i<globalSize ;i++){
			sum2=sum2+((float)temp8[i]/multR+minR);
		}
		tempOutput->unMap_ZeroCopyObject_ReadFlagHalf(temp6);
		outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlag8Bit(temp8);
		LOGI("----end b=======");
	}else{
		half* temp8;
		temp8=outputWrapper8Bit->map_ZeroCopyObject_ReadFlagHalf();
		temp6=tempOutput->map_ZeroCopyObject_ReadFlagHalf();
		LOGI("globalSize=%d %d",globalSize,batchSize * dim.outputCubeSize);
		for(int i=0;i<10 ;i++){
			LOGI("-----------------------[%f vs %f",HalfToFloat(temp6[i]),HalfToFloat(temp8[i]));
		}
		for(int i=0;i<globalSize ;i++){

			error=error+abs(HalfToFloat(temp6[i])-HalfToFloat(temp8[i]));
		}
		for(int i=0;i<globalSize ;i++){
		sum=sum+HalfToFloat(HalfToFloat(temp6[i]));
		}
		for(int i=0;i<globalSize ;i++){
		sum2=sum2+HalfToFloat(HalfToFloat(temp8[i]));
		}
		tempOutput->unMap_ZeroCopyObject_ReadFlagHalf(temp6);
		outputWrapper8Bit->unMap_ZeroCopyObject_ReadFlagHalf(temp8);
	}


LOGI("------------------sum %f vs %f size %d %d %d error %f",sum,sum2,batchSize * dim.outputCubeSize,globalSize,workgroupsize,error/globalSize);
LOGI("------------------");

#endif

cl->finish();
#endif

}


VIRTUAL void Forward1::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
#if TEST_FORWARD==1

	float * output=(float*)outputWrapper->getHostArray();

if (false/*dim.useHalfMemory*/){


	////////////////////////////
		biasWrapper->copyToHost();
		float *bias= (float *)biasWrapper->getHostArray();
			half *biasHalf=new half[batchSize * dim.outputCubeSize];
		for (int i=0; i<batchSize * dim.outputCubeSize;i++)
			biasHalf[i] = FloatToHalf(bias[i]);
		biasWrapper->copyToDevice();
		CLWrapper *biasHalfWrapper = (CLWrapper *)cl->wrap(batchSize * dim.outputCubeSize, biasHalf);
		biasHalfWrapper->createOnDevice();
		biasHalfWrapper->copyToDevice();
	/////
	dataWrapper->copyToHost();
	float *data= (float *)dataWrapper->getHostArray();
		half *dataHalf=new half[dim.inputCubeSize*batchSize];
		for (int i=0; i<dim.inputCubeSize*batchSize;i++)
			dataHalf[i] = FloatToHalf(data[i]);

		CLWrapper *dataHalfWrapper = (CLWrapper *)cl->wrap(dim.inputCubeSize*batchSize, dataHalf);
		dataHalfWrapper->createOnDevice();
		dataHalfWrapper->copyToDevice();

///////////
		float *weights= (float *)weightsWrapper->getHostArray();
		half *weightsHalf=new half[dim.inputPlanes*dim.filterSizeSquared*dim.numFilters];
		for (int j=0; j<dim.numFilters;j++)
			for (int i=0; i<dim.inputPlanes*dim.filterSizeSquared;i++)
				if (i<dim.inputPlanes*dim.filterSizeSquared)
					weightsHalf[j*dim.inputPlanes*dim.filterSizeSquared+i] = FloatToHalf(weights[dim.filterSizeSquared*dim.inputPlanes*j+i]);
				else
					weightsHalf[j*dim.inputPlanes*dim.filterSizeSquared+i] = FloatToHalf(0.0);
		CLWrapper *weightsHalfWrapper = (CLWrapper *)cl->wrap(/*dim.filterSizeSquared**/dim.inputPlanes*dim.filterSizeSquared*dim.numFilters/**dim.inputPlanes*/, weightsHalf);
		weightsHalfWrapper->createOnDevice();
		weightsHalfWrapper->copyToDevice();
///////////

	//////////////
	half *outputHalf=new half[batchSize * dim.outputCubeSize];
	for (int i=0; i<batchSize * dim.outputCubeSize;i++)
		outputHalf[i] = FloatToHalf(0.0f);
	CLWrapper *outputHalfWrapper= (CLWrapper *)cl->wrap(batchSize * dim.outputCubeSize,outputHalf);
	LOGI("3");
	outputHalfWrapper->createOnDevice();
	outputHalfWrapper->copyToDevice();
	//////////////

	clock_t startTimer2, stopTimer2;
	startTimer2=clock();
	StatefulTimer::timeCheck("Forward1::forward START");

	kernelH->in(batchSize);
	kernelH->input(dataHalfWrapper);
	kernelH->input(weightsHalfWrapper);
	kernelH->output(outputHalfWrapper);
	int globalSize2 = batchSize * dim.outputCubeSize;
	int workgroupsize2 = kernelH->get_kernel_work_group_size();
	LOGI("workgroupsize2 %d, workgroupsize2 %d",workgroupsize2, cl->getMaxWorkgroupSize());
	globalSize2 = (( globalSize2 + workgroupsize2 - 1) / workgroupsize2) * workgroupsize2;
	kernelH->run_1d(globalSize2, workgroupsize2);
	cl->finish();


	stopTimer2 = clock();
		double elapse2 = 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC;
		LOGI("convolution half took %g ms\n\n", 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC) ;

	if(dim.biased) {


	        addBias->forward2(
	            batchSize, dim.numFilters, dim.outputSize,
	            outputHalfWrapper, biasHalfWrapper);
	        outputHalfWrapper->copyToHost();
	        half * convh=(half*)outputHalfWrapper->getHostArray();
	        	for (int i =0;i<batchSize * dim.outputCubeSize;i++){
	        				output[i]=HalfToFloat(convh[i]);
	        	}
	        	outputWrapper->copyToDevice();
	    }


	StatefulTimer::timeCheck("Forward1::forward after call forward");


	//outputWrapper->copyToHost();

//    LOGI("////////////conv////////");
//
//    float * conv=(float*)outputWrapper->getHostArray();
//    int col=dim.outputSize;
//    float sum=0.0f;
//    for(int i =0;i<batchSize * dim.outputCubeSize; i++){
//        sum+=abs(temp[i]-conv[i]);
//    }
//    LOGI("diff %f",sum);

//    for(int i =0;i<10; i++){
//    	LOGI("temp[i]=%f conv[i]=%f ",temp[i],conv[i]);
//    }
    delete biasHalfWrapper;
    delete [] biasHalf;
	delete dataHalfWrapper;
	delete[] dataHalf;
	delete weightsHalfWrapper;
	delete[] weightsHalf;
	delete outputHalfWrapper;
	delete[] outputHalf;
}else{

	///////////////////////

	clock_t startTimer1, stopTimer1;
	startTimer1=clock();
    StatefulTimer::timeCheck("Forward1::forward START");
    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
LOGI("dim.outputCubeSize %d",dim.outputCubeSize);
    int globalSize = batchSize * dim.outputCubeSize;
    int workgroupsize = kernel->get_kernel_work_group_size();
    //int workgroupsize = std::min(globalSize, cl->getMaxWorkgroupSize());
    LOGI("workgroupsize %d, workgroupsize2 %d",workgroupsize, cl->getMaxWorkgroupSize());
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;

    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("Forward1::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("Forward1::forward END");



stopTimer1 = clock();
double elapse = 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC;
LOGI("convolution took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

outputWrapper->copyToHost();

    	    float * conv=(float*)outputWrapper->getHostArray();
int col=dim.outputSize;

LOGI("2. dim.outputCubeSize %d",dim.outputCubeSize);
LOGI("2. dim.outputSizeSquared %d",dim.outputSizeSquared);

LOGI("col %d",dim.outputSize);
    	    LOGI("////////////conv////////");
    	    for (int i =0;i<10;i++){
    			string displayArraY="";
    			for (int j =0;j<10;j++){
    				displayArraY= displayArraY+ "-" + to_string_with_precision2(conv[i*col+j+1*dim.outputCubeSize]);
    			}
    			LOGI("%s",displayArraY.c_str());
    			displayArraY.clear();
    	    }
    	    LOGI("////////////conv////////");
//    	    float sum=0.0f;
//    	    for(int i =0;i<batchSize * dim.outputCubeSize; i++){
//    	    	//LOGI("temp[i]=%f conv[i]=%f ",temp[i],conv[i]);
//    	        sum+=abs(temp[i]-conv[i]);
//    	    }
//    	    LOGI("diff %f",sum);
}
//delete [] temp;
#endif
}

Forward1::Forward1(bool needToNormalize,int batchSize,EasyCL *cl, LayerDimensions dim)
        {

	this->cl=cl;
	this->dim=dim;

	normalization=needToNormalize;
	setup=false;
	setupHalf=false;
	setupEightBit=false;
	#if TEST_FORWARD==1
    	addBias = new AddBias(cl);
    	LOGI("2)numFilters %d",dim.numFilters);

    std::string options = "";
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward1.cl", "convolve_imagecubes_float2", 'options')
    // ]]]
    // generated using cog, from cl/forward1.cl:
    const char * kernelSource =
    	    "void kernel convolve_imagecubes_float2(\n"
    	    "    const int numExamples,\n"
    	    "      global const float *inputs, global const float *filters,\n"
    	    "    global float *output) {\n"
    	    "    int globalId = get_global_id(0);\n"
    	    "\n"
    	    "    int exampleId = (globalId / gOutputSizeSquared) / gNumFilters;\n"
    	    "    int filterId = (globalId / gOutputSizeSquared) % gNumFilters;\n"
    	    "\n"
    	    "    // intraimage coords\n"
    	    "    int outputRow = (globalId % gOutputSizeSquared) / gOutputSize;\n"
    	    "    int outputCol = (globalId % gOutputSizeSquared) % gOutputSize;\n"
    	    "\n"
    	    "\n"
    	    "    float sum = 0;\n"
    	    "    if (exampleId < numExamples) {\n"
    	    "        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {\n"
    	    "            global float const*inputPlane = inputs + exampleId * gNumInputPlanes * gInputSizeSquared + inputPlaneIdx * gInputSizeSquared;\n"
    	    "            global float const*filterPlane = filters + filterId * gNumInputPlanes * gFilterSizeSquared + inputPlaneIdx * gFilterSizeSquared;\n"
    	    "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    	    "                // trying to reduce register pressure...\n"
    	    "                #if gPadZeros == 1\n"
    	    "                    #define inputRowIdx (outputRow + u)\n"
    	    "                #else\n"
    	    "                    #define inputRowIdx (outputRow + u + gHalfFilterSize)\n"
    	    "                #endif\n"
    	    "                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;\n"
    	    "                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    	    "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;\n"
    	    "                #pragma unroll\n"
    	    "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    	    "                    #if gPadZeros == 1\n"
    	    "                        #define inputColIdx (outputCol + v)\n"
    	    "                    #else\n"
    	    "                        #define inputColIdx (outputCol + v + gHalfFilterSize)\n"
    	    "                    #endif\n"
    	    "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;\n"
    	    "                    if (process) {\n"
    	    "                            sum += inputRow[inputColIdx] * filterRow[v];\n"
    	    "                    }\n"
    	    "                }\n"
    	    "            }\n"
    	    "        }\n"
    		"		output[globalId] = sum;\n"
    	    "    }\n"
    	    "\n"
    	    "}\n"
    	    "\n"
    	    "";

    string operation="Forward1_"+std::to_string(dim.numFilters);
    kernel = cl->buildKernelFromString(operation, kernelSource, "convolve_imagecubes_float2", options, "../../cl/forward1.cl");


    const char * kernelSource2 =
    				"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
    		        "void kernel convolve_imagecubes_half2(\n"
    		        "    const int numExamples,\n"
    		        "      global const half *inputs, global const half *filters,\n"
    		        "    global half *output) {\n"
    		        "    int globalId = get_global_id(0);\n"
    		        "\n"
    		        "    int outputImage2Id = globalId / gOutputSizeSquared;\n"
    		        "    int exampleId = outputImage2Id / gNumFilters;\n"
    		        "    int filterId = outputImage2Id % gNumFilters;\n"
    		        "\n"
    		        "    // intraimage coords\n"
    		        "    int localid = globalId % gOutputSizeSquared;\n"
    		        "    int outputRow = localid / gOutputSize;\n"
    		        "    int outputCol = localid % gOutputSize;\n"
    		        "\n"
    		        "    global half const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;\n"
    		        "    global half const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;\n"
    		        "\n"
    		        "    half sum = convert_half(0.0f)\n"
    		        "    if (exampleId < numExamples) {\n"
    		        "        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {\n"
    		        "            global half const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;\n"
    		        "            global half const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;\n"
    		        "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    		        "                // trying to reduce register pressure...\n"
    		        "                #if gPadZeros == 1\n"
    		        "                    #define inputRowIdx (outputRow + u)\n"
    		        "                #else\n"
    		        "                    #define inputRowIdx (outputRow + u + gHalfFilterSize)\n"
    		        "                #endif\n"
    		        "                global half const *inputRow = inputPlane + inputRowIdx * gInputSize;\n"
    		        "                global half const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    		        "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;\n"
    		        "                #pragma unroll\n"
    		        "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    		        "                    #if gPadZeros == 1\n"
    		        "                        #define inputColIdx (outputCol + v)\n"
    		        "                    #else\n"
    		        "                        #define inputColIdx (outputCol + v + gHalfFilterSize)\n"
    		        "                    #endif\n"
    		        "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;\n"
    		        "                    if (process) {\n"
    		        "                            sum += inputRow[inputColIdx] * filterRow[v];\n"
    		        "                    }\n"
    		        "                }\n"
    		        "            }\n"
    		        "        }\n"
    		        "    }\n"
    		        "\n"
    		        "    if (exampleId < numExamples) {\n"
    		        "        output[globalId] = sum;\n"
    		        "    }\n"
    		        "}\n"
    		        "\n"
    		        "";
        string operation2="ForwardHalf1_"+std::to_string(dim.numFilters);

        options=options+" -qcom-accelerate-16-bit -qcom-sched-rule=2";
        LOGI("option: %s ",options.c_str());
        kernelH = cl->buildKernelFromString(operation2, kernelSource2, "convolve_imagecubes_half2", options, "../../cl/forward1.cl");
#endif

//        string operation3="testForward1_"+std::to_string(dim.numFilters);
//    test = cl->buildKernelFromString(operation3, kernelSource, "convolve_imagecubes_float2", options, "../../cl/forward1.cl");
        buildKernelConvolve(batchSize);
}

void Forward1::buildKernelConvolve(int batchSize) {
	ConfigManager binariesManager( cl->absolutePath/*path,binaryFileRep*/);
#if EIGHT_BIT_ACCURACY ==0
	#if HALF_ACCURACY==0
		TemplatedKernel builder(cl);
		 //string identifier2="testForward1_"+std::to_string(dim.numFilters);

		string identifier2="testForward1_";
		#if TRANSFER ==0
			identifier2=identifier2+"NormalTraining";
		#else
			identifier2=identifier2+"TransferLearninng";
		#endif
			 identifier2=identifier2+"nbFilter=";
			 identifier2=identifier2+std::to_string(dim.numFilters);
			 identifier2=identifier2+"_InputSize="+std::to_string(dim.inputSize);
			 identifier2=identifier2+"_batchsize="+std::to_string(dim.batchsize);
			 identifier2=identifier2+"_OutputSize="+std::to_string(dim.outputSize);
			 identifier2=identifier2+"_conv="+BoolToString(dim.isConv);
			 identifier2=identifier2+"_normalize="+BoolToString(dim.needToNormalize);
			 identifier2=identifier2+"_maxpool="+BoolToString(dim.useMaxPooling);



		bool compiled;
		string filepath="default";
		if (not binariesManager.alreadyCompiledKernel("convolve_imagecubes_float2","",identifier2,filepath)){


			setupBuilderConvolve(&builder, batchSize);
		}

			this->test = builder.buildKernel(
					identifier2,
				   "forward1.cl",
				   getKernelTemplateConvolve(),
				   "convolve_imagecubes_float2",
				   false
			);
			////////////////////////
	#else
		TemplatedKernel builderHalf(cl);

		string identifier3="testForward1_Half_";
	//	#if TRANSFER ==0
	//		identifier3=identifier3+"NormalTraining";
	//	#else
	//		identifier3=identifier3+"TransferLearninng";
	//	#endif
		identifier3=identifier3+"nbFilter=";
		identifier3=identifier3+std::to_string(dim.numFilters);
		identifier3=identifier3+"_InputSize="+std::to_string(dim.inputSize);
		identifier3=identifier3+"_batchsize="+std::to_string(dim.batchsize);
		identifier3=identifier3+"_OutputSize="+std::to_string(dim.outputSize);
		identifier3=identifier3+"_conv="+BoolToString(dim.isConv);
		identifier3=identifier3+"_normalize="+BoolToString(dim.needToNormalize);
		identifier3=identifier3+"_maxpool="+BoolToString(dim.useMaxPooling);


		//ConfigManager*binariesManager=new ConfigManager( cl->absolutePath/*path,binaryFileRep*/);
		//bool compiled;
		string filepath2="default";
		if (not binariesManager.alreadyCompiledKernel("convolve_imagecubes_float2Half","",identifier3,filepath2)){


			setupBuilderConvolveHalf(&builderHalf, batchSize);
		}


		this->test2 = builderHalf.buildKernel(
						identifier3,
					   "forward1.cl",
					   getKernelTemplateConvolveHalf(),
					   "convolve_imagecubes_float2Half",
					   false
				);
	#endif

	///////////////////////////
#else

#if 0
    ExtractMinMultFromFile(minW, multW,minI, multI,minR, multR);
#else
    minW=1.0;
    multW=1.0;
    minI=1.0;
    multI=1.0;
    minR=1.0;
    multR=1.0;
#endif
    dim.multI=multI;
    dim.minI=minI;
	if (not cl->fit8BitData){
		string identifier4=" ";
		TemplatedKernel builderEightBit(cl);
		if (not dim.isLast){
			identifier4="testForward1_EightBit_notLast";
//			#if TRANSFER ==0
//						identifier4=identifier4+"NormalTraining";
//			#else
//						identifier4=identifier4+"TransferLearninng";
//			#endif
			identifier4=identifier4+"nbFilter=";
			identifier4=identifier4+std::to_string(dim.numFilters);
			identifier4=identifier4+"_InputSize="+std::to_string(dim.inputSize);
			identifier4=identifier4+"_batchsize="+std::to_string(dim.batchsize);
			identifier4=identifier4+"_OutputSize="+std::to_string(dim.outputSize);
			identifier4=identifier4+"_conv="+BoolToString(dim.isConv);
			identifier4=identifier4+"_normalize="+BoolToString(dim.needToNormalize);
			identifier4=identifier4+"_maxpool="+BoolToString(dim.useMaxPooling);
//			identifier4=identifier4+"_multR="+std::to_string(multR);
//			identifier4=identifier4+"_minR="+std::to_string(minR);
			identifier4=identifier4+"_multW="+std::to_string(multW);
			identifier4=identifier4+"_minW="+std::to_string(minW);
			identifier4=identifier4+"_multI="+std::to_string(multI);
			identifier4=identifier4+"_minI="+std::to_string(minI);
		}else{
#if HALF_ACCURACY==0
			identifier4="testForward1_EightBit_Last";
//			#if TRANSFER ==0
//						identifier4=identifier4+"NormalTraining";
//			#else
//						identifier4=identifier4+"TransferLearninng";
//			#endif
			identifier4=identifier4+"nbFilter=";
			identifier4=identifier4+std::to_string(dim.numFilters);
			identifier4=identifier4+"_InputSize="+std::to_string(dim.inputSize);
			identifier4=identifier4+"_batchsize="+std::to_string(dim.batchsize);
			identifier4=identifier4+"_OutputSize="+std::to_string(dim.outputSize);
			identifier4=identifier4+"_conv="+BoolToString(dim.isConv);
			identifier4=identifier4+"_normalize="+BoolToString(dim.needToNormalize);
			identifier4=identifier4+"_maxpool="+BoolToString(dim.useMaxPooling);
//			identifier4=identifier4+"_multR="+std::to_string(multR);
//			identifier4=identifier4+"_minR="+std::to_string(minR);
			identifier4=identifier4+"_multW="+std::to_string(multW);
			identifier4=identifier4+"_minW="+std::to_string(minW);
			identifier4=identifier4+"_multI="+std::to_string(multI);
			identifier4=identifier4+"_minI="+std::to_string(minI);
#else
			identifier4="testForward1_EightBit_LastHalf";
//			#if TRANSFER ==0
//						identifier4=identifier4+"NormalTraining";
//			#else
//						identifier4=identifier4+"TransferLearninng";
//			#endif
			identifier4=identifier4+"nbFilter=";
			identifier4=identifier4+std::to_string(dim.numFilters);
			identifier4=identifier4+"_InputSize="+std::to_string(dim.inputSize);
			identifier4=identifier4+"_batchsize="+std::to_string(dim.batchsize);
			identifier4=identifier4+"_OutputSize="+std::to_string(dim.outputSize);
			identifier4=identifier4+"_conv="+BoolToString(dim.isConv);
			identifier4=identifier4+"_normalize="+BoolToString(dim.needToNormalize);
			identifier4=identifier4+"_maxpool="+BoolToString(dim.useMaxPooling);
//			identifier4=identifier4+"_multR="+std::to_string(multR);
//			identifier4=identifier4+"_minR="+std::to_string(minR);
			identifier4=identifier4+"_multW="+std::to_string(multW);
			identifier4=identifier4+"_minW="+std::to_string(minW);
			identifier4=identifier4+"_multI="+std::to_string(multI);
			identifier4=identifier4+"_minI="+std::to_string(minI);
#endif
		}

			string filepath3="default";
			if (not binariesManager.alreadyCompiledKernel("convolve_imagecubes_float2EightBit","",identifier4,filepath3)){


				setupBuilderConvolveEightBit(&builderEightBit, batchSize);
				LOGI("0");
			}


				this->test4 = builderEightBit.buildKernel(
								identifier4,
							   "forward1.cl",
							   getKernelTemplateConvolveEightBit(dim),
							   "convolve_imagecubes_float2EightBit",
							   false
						);

	}
	LOGI("2");
	#endif
}




void Forward1::setupBuilderConvolve(TemplatedKernel *builder,int batchSize) {

	string activationFunction("linear");
	string partialVectorizationType="default";
	string partialVectorizationLoad="default";
	string constantMemPartialVectorizationLoad="default";
	string initializationCondition="default";
	string internalLoopString1="default";
	string internalLoopString1norm="default";
	string internalLoopString2="default";
	string internalLoopStringNormalization="default";
	string internalLoopString1withPartialVectorization="default";
	bool fullvectorization=true;
	bool partialvectorization=true;
	bool ok1=true;
	int loop_count_partialVectorization=0;
	int remainerPartialVectorization=0;
	string initString="default";
	string dotString="default";
	string loop_string_partialVectorization="default";
	string extra_loop_string_partialVectorization="default";
	int vectorSize=0;
    string outputPoolingSelectorString="";
	string endPoolingString="}\n";
	string endPoolingString2="";
	string poolingSelectorString="";

	if (dim.useMaxPooling){
		setPoolingLayer(outputPoolingSelectorString,endPoolingString,endPoolingString2,poolingSelectorString,builder);
	}else{
		builder->set("gresult", "{{gActivationFunction}}");
		setActivationFunction(builder);
		setNonPoolingLayerVariable(builder,endPoolingString,endPoolingString2,fullvectorization);
	}


	testCondition(ok1);


	int countVectorizationPadding=dim.inputPlanes%4;
	if (countVectorizationPadding!=0)
		fullvectorization=false;


	if (partialvectorization){
		setAutoVectorization(vectorSize,remainerPartialVectorization,loop_count_partialVectorization,ok1, partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad,initializationCondition,builder,loop_string_partialVectorization, extra_loop_string_partialVectorization, initString, dotString);
	}

	setHintCompiler(batchSize,fullvectorization,partialvectorization,partialVectorizationType,builder);

//	if ((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0)))
//	if (normalization)
//		if (partialvectorization)
//			LOGI("internalLoopStringNormalization");
//		else
//			LOGI("internalLoopString1norm");
//	else
//		if(partialvectorization)
//			LOGI("internalLoopString1withPartialVectorization");
//		else
//			LOGI("internalLoopString1");
//	else
//		LOGI("internalLoopString2");

	setInternalLoop(ok1,loop_count_partialVectorization,internalLoopString1,internalLoopString1norm,internalLoopString2,internalLoopStringNormalization,internalLoopString1withPartialVectorization,initializationCondition,loop_string_partialVectorization,extra_loop_string_partialVectorization,partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad);

	writeKernelcode(builder, outputPoolingSelectorString,  poolingSelectorString,  partialvectorization,  normalization,  internalLoopStringNormalization,  internalLoopString1norm,  internalLoopString1withPartialVectorization,  internalLoopString1,  internalLoopString2, fullvectorization,  batchSize, ok1);


}

STATIC std::string Forward1::getKernelTemplateConvolve() {


	const char * kernelSource =
	    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2(\n"
	    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
	    	    "    global float *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
	    	    "    const int globalId = get_global_id(0);\n"
				"{{SelectoBegin}}\n"
	    	    "\n"
	    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
	    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
	    	    "\n"
				"    {{gVectorType}} sum = 0;\n"
				"    {{gMaxPooling}}"
	    	    "        {{gImageRowCoord}}"
	    	    "        {{gImageColCoord}}"
	    	    "        {{gBeginFirstLoop}}\n"
	    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
				"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
	    		"        {{gInternalLoop}}"
	    	    "      {{gEndFirstLoop}}\n"
				"\n"
				"    {{gMaxPoolingEnd}}\n"
	    		"    output[globalId] = {{gresult}};\n"
				"    {{outputPoolingSelector}}"
	    	    "\n"
				"{{SelectoEnd}}"
	    	    "}\n"
	    	    "\n"
	    	    "";



    return kernelSource;
}



void Forward1::testCondition(bool &ok1){
	if(dim.outputSizeSquared==1){
			for (int u = -dim.halfFilterSize; u <= (dim.halfFilterSize-(dim.filterSize % 2 == 0 ? 1 : 0)); u++){
				int temp = u+dim.halfFilterSize;
				if ((temp < 0 ) || (temp >= dim.inputSize)){
					ok1=false;
					break;
				}
			}
	}else
	ok1=false;
}


void Forward1::setNonPoolingLayerVariable(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization){

	builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
	builder->set("gSumAndBias", dim.biased? "{{gSum}}+bias[(globalId {{DIVgOutputSizeSquared}}) % {{gNumFilters}} ]":"{{gSum}}");
	builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((globalId % {{gOutputSizeSquared}}) % {{gOutputSize}})*"+to_string(dim.stride)+";\n":"int outputCol = (globalId % {{gOutputSizeSquared}}) % {{gOutputSize}};\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = ((globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}})*"+to_string(dim.stride)+";\n":"int outputRow = (globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}};\n":"");
	builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.outputSize):"");
	builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
	builder->set("gSum", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "dot( sum,(float4)(1.0f,1.0f,1.0f,1.0f))":"sum"));
	builder->set("gMaxPooling", "");
	builder->set("gMaxPoolingEnd", "");
}


void  Forward1::setActivationFunction(TemplatedKernel *builder){


		if (dim.activationLayer==1)
			builder->set("gActivationFunction", "{{gSumAndBias}}");
		if (dim.activationLayer==2)
			builder->set("gActivationFunction", "fmax ( {{gSumAndBias}} , 0 )");
		if (dim.activationLayer==3)
			builder->set("gActivationFunction", "tanh ( {{gSumAndBias}} )");
		if (dim.activationLayer==4)
			builder->set("gActivationFunction", "(1.7159f * tanh(0.66667f * {{gSumAndBias}}))");
		if (dim.activationLayer==5)
			builder->set("gActivationFunction", "(1.0f / (1 + exp(- ({{gSumAndBias}}))))");
		if (dim.activationLayer==6)
			builder->set("gActivationFunction", "fmin (fmax ( {{gSumAndBias}} , 0 ),(exp({{gSumAndBias}}) - 1))");

}


void Forward1::setPoolingLayer(    string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder){
#if TRANSFER ==0
	poolingSelectorString=", global int * selectorArray, global float *gradInput";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunction(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="float maxPool=-999.99f;\n"
								"    int selectorID=100;\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunction(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="float maxPool=-999.99f;\n"
									   "    int selectorID=100;\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}
#else
		poolingSelectorString="";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunction(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="float maxPool=-999.99f;\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString=";\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunction(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="float maxPool=-999.99f;\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="\n";
	}
#endif
}
void Forward1::setAutoVectorization(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString){

		vector<string>indexOpencl;
		indexOpencl.push_back("0");
		indexOpencl.push_back("1");
		indexOpencl.push_back("2");
		indexOpencl.push_back("3");
		indexOpencl.push_back("4");
		indexOpencl.push_back("5");
		indexOpencl.push_back("6");
		indexOpencl.push_back("7");
		indexOpencl.push_back("8");
		indexOpencl.push_back("9");
		indexOpencl.push_back("a");
		indexOpencl.push_back("b");
		indexOpencl.push_back("c");
		indexOpencl.push_back("d");
		indexOpencl.push_back("e");
		indexOpencl.push_back("f");
		int size =dim.filterSize;
		int cpt=0;
		vectorSize=4;
		partialVectorizationType="float4";
		partialVectorizationLoad="(*((__global float4*)&";
		constantMemPartialVectorizationLoad="(*((__constant float4*)&";
		initString="(float4)(0.0f,0.0f,0.0f,0.0f)";
		dotString="(float4)(1.0f,1.0f,1.0f,1.0f)";

		loop_count_partialVectorization=floor((size)/4);
		remainerPartialVectorization=floor((size)%4);
		if ((not ok1)&&(loop_count_partialVectorization==1)){
			cpt=0;
			initializationCondition="";
			for(int i=-(dim.filterSize >> 1);i<(vectorSize-(dim.filterSize >> 1));i++){
				initializationCondition+="            conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"),0,1)));\n";
				cpt++;
			}
		}


if (loop_count_partialVectorization>=1){

		if ((not ok1)&&(loop_count_partialVectorization!=1)){
			initializationCondition="";

				loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
				loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(vectorSize);i++){
						loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}

				if (normalization)
					loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
				else loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

				//loop_string_partialVectorization+= "               sum=dot( inputsV*filterV,"+dotString+");\n            }";
		}else
			if (ok1){

				if ((remainerPartialVectorization)<4){
					int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
					initializationCondition="";
					extra_loop_string_partialVectorization="";
					if (loop_count_partialVectorization>1){
						loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
						loop_string_partialVectorization+="              float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
														 "              float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

						loop_string_partialVectorization+= "              sum+=dot( (inputsV*filterV),"+dotString+");\n            }\n            ";
					}else{

						loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "sum+=dot( (inputsV*filterV),"+dotString+");\n";

					}

					if (remainerPartialVectorization!=0){

						if ((ok1)&&((dim.filterSize >> 1)==1)){
							extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";

							extra_loop_string_partialVectorization+="             float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																 "             float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
								else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
							}else{
								extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
/////////////////////////
								if (loop_count_partialVectorization>1){
									extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}else{
									extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}
/////////////////////////
								//extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
								//									 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
								else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
							}
						}
				}else{

					initializationCondition="";
					extra_loop_string_partialVectorization="";
					loop_string_partialVectorization= "sum+=dot( (inputsV*filterV),"+dotString+");\n";

					if (remainerPartialVectorization!=0){
						extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
						extra_loop_string_partialVectorization+="          for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
							extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
															 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

							cpt=0;
							for(int i=0;i<(remainerPartialVectorization);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
									cpt++;
								}
							for(int i=remainerPartialVectorization;i<(vectorSize);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
									cpt++;
								}

							if (normalization)
								extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
							else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

						}
				}


			}else{

				if (normalization){
					loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
				}
				else{
					loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
				}
			}

		if ((not ok1)&&((remainerPartialVectorization)!=0)){


			if ((remainerPartialVectorization)<4){

				int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
				if (loop_count_partialVectorization>1){
					extra_loop_string_partialVectorization="           float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}else{
					extra_loop_string_partialVectorization="            inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
												 "            filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
				else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";

			}else{

				extra_loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
				extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
				else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

			}
		}
	}else{

			initializationCondition="";//no need
			loop_string_partialVectorization="";//no need
			int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
			extra_loop_string_partialVectorization="           float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
			cpt=0;
			for(int i=0;i<(remainerPartialVectorization);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
				cpt++;
			}
			for(int i=remainerPartialVectorization;i<(vectorSize);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
				cpt++;
			}

			if (normalization)
				extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
			else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";

		}
	}


void Forward1::setHintCompiler(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder){
	int possibleGlobalSize = batchSize * dim.outputCubeSize;
	int possibleWorkgroupsize = std::min(possibleGlobalSize, cl->getMaxWorkgroupSize());

	string hintCompilerString="__attribute__((vec_type_hint(";
	if ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
		hintCompilerString+="float4";
	else{
		if ((not fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
			hintCompilerString+="float";
		else
			if (partialvectorization)
				hintCompilerString+=partialVectorizationType;
			else hintCompilerString+="float";
	}

	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);
}

void Forward1::setInternalLoop(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad){

	internalLoopString1="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += inputs[inputRow + {{inputColIdx}}] * filters[filterRowIdx+v] {{gCondition}};\n"
								"            }\n"
								"        }\n";

	internalLoopString1norm="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += scale*(inputs[inputRow + {{inputColIdx}}]+translate) * filters[filterRowIdx+v] {{gCondition}};\n"
								"            }\n"
								"        }\n";

	internalLoopString2="sum += inputs[inputPlaneIdx] * filters[filterIdx] {{gCondition}};\n";

	internalLoopStringNormalization="";
	if (not ok1)
		internalLoopStringNormalization+=partialVectorizationType+" conditionVector;\n";
	if (loop_count_partialVectorization!=1){

		internalLoopStringNormalization+="#pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}else{
		internalLoopStringNormalization+="#pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}
	internalLoopString1withPartialVectorization="";
	if (not ok1)
		internalLoopString1withPartialVectorization+=partialVectorizationType+" conditionVector;\n";
	if ((ok1)&&((dim.filterSize >> 1)==1)){
		internalLoopString1withPartialVectorization+="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}else{

	internalLoopString1withPartialVectorization+="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+loop_string_partialVectorization+""
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}
}

void Forward1::writeKernelcode(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString2,bool fullvectorization, int batchSize,bool ok1){

	builder->set("outputPoolingSelector", outputPoolingSelectorString);
	builder->set("gPoolingOutputSelector", poolingSelectorString);

	builder->set("gNormalization", normalization? "    ,\n float translate, float scale":"");
	builder->set("gVectorType",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "float4":"float"));

	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gInternalLoop",((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)))? normalization? partialvectorization? internalLoopStringNormalization:internalLoopString1norm:partialvectorization? internalLoopString1withPartialVectorization:internalLoopString1:internalLoopString2);


	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gPlusInputPlaneIdxTimeGInputSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gInputSizeSquared}}");

	builder->set("gBeginFirstLoop",(dim.inputPlanes==1)? "":((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "      #pragma unroll\n    for (int planeId = 0; planeId < "+to_string((dim.inputPlanes/4))+"; planeId++) {\n":"      #pragma unroll\n    for (int planeId = 0; planeId < {{gNumInputPlanes}}; planeId++) {\n"));
	builder->set("gCondition", ok1 ? "":(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0))? "*((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputColIdx}},0,1)))":"");
	builder->set("gHalfFilterSizeMinusGEven", (dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0));
	builder->set("gNumInputPlanesTimeGFilterSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? (dim.inputPlanes*dim.filterSizeSquared/4):dim.inputPlanes*dim.filterSizeSquared));
	builder->set("gNumInputPlanesTimeGInputSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? ((dim.inputPlanes*dim.inputSizeSquared)/4):dim.inputPlanes*dim.inputSizeSquared));
	builder->set("gLimit", batchSize * dim.numFilters * dim.outputSize * dim.outputSize);

	builder->set("gBias", dim.biased? ", __constant  float * bias":" ");
	builder->set("gNumExamples", batchSize);
	builder->set("inputRowIdx", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputRowIdx2", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputColIdx",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol + v)" : "(outputCol + v + {{gHalfFilterSize}})": dim.padZeros ? "(v)" : "(v + {{gHalfFilterSize}})");
	builder->set("inputColIdx2",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol)" : "(outputCol + {{gHalfFilterSize}})": dim.padZeros ? "" : "({{gHalfFilterSize}})");
    builder->set("gNumInputPlanes", dim.inputPlanes);
    builder->set("gInputPlanes", dim.inputPlanes);
    builder->set("gInputSize", dim.inputSize);
    builder->set("gInputSizeSquared", dim.inputSizeSquared);
    builder->set("gNumFilters", dim.numFilters);
    builder->set("gFilterSize", dim.filterSize);
    builder->set("gHalfFilterSize",  dim.filterSize >> 1);
    builder->set("gFilterSizeSquared", dim.filterSizeSquared);
    builder->set("gNumOutputPlanes", dim.numFilters);
    builder->set("gOutputPlanes", dim.numFilters);
	builder->set("gOutputSize", dim.outputSize);
    builder->set("gOutputSizeSquared", dim.outputSizeSquared);
    builder->set("gPadZeros", dim.padZeros ? 1 : 0);
    builder->set("gMargin", dim.padZeros ? dim.filterSize >> 1 : 0);
    builder->set("gEven", dim.filterSize % 2 == 0 ? 1 : 0);
    builder->set("gSkip", dim.skip);

//	float globalSize;
//	float workgroupsize;
//	if (dim.useMaxPooling)
//			globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
//	else
//		globalSize = batchSize * dim.outputCubeSize;
//	workgroupsize = 1024;//test->get_kernel_work_group_size();test->get_kernel_work_group_size();
//	globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
//
//	if (dim.useMaxPooling){
//		builder->set("SelectoEnd",(globalSize> batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput)? "}" :"");
//		builder->set("SelectoBegin",(globalSize> batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput)? "if ( get_global_id(0)<"+to_string(batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput)+"){\n" :"");
//	}else{
//		builder->set("SelectoEnd",(globalSize> batchSize * dim.outputCubeSize)? "}":"");
//		builder->set("SelectoBegin",(globalSize> batchSize * dim.outputCubeSize)? "if ( get_global_id(0)<"+to_string(batchSize * dim.outputCubeSize)+"){\n" :"");
//	}

}

///////////////////////

STATIC std::string Forward1::getKernelTemplateConvolveHalf() {


	const char * kernelSource =
			    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
	    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2Half(\n"
	    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
	    	    "    global half *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
	    	    "    const int globalId = get_global_id(0);\n"
				"{{SelectoBegin}}\n"
	    	    "\n"
	    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
	    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
	    	    "\n"
				"    {{gVectorType}} sum = {{gVectorTypeInitialization}};\n"
				"    {{gMaxPooling}}"
	    	    "        {{gImageRowCoord}}"
	    	    "        {{gImageColCoord}}"
	    	    "        {{gBeginFirstLoop}}\n"
	    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
				"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
	    		"        {{gInternalLoop}}"
	    	    "      {{gEndFirstLoop}}\n"
				"\n"
				"    {{gMaxPoolingEnd}}\n"
	    		"    output[globalId] = {{gresult}};\n"
				"    {{outputPoolingSelector}}"
	    	    "\n"
				"{{SelectoEnd}}"
	    	    "}\n"
	    	    "\n"
	    	    "";




    return kernelSource;
}



void Forward1::setupBuilderConvolveHalf(TemplatedKernel *builder,int batchSize) {

	string activationFunction("linear");
	string partialVectorizationType="default";
	string partialVectorizationLoad="default";
	string constantMemPartialVectorizationLoad="default";
	string initializationCondition="default";
	string internalLoopString1="default";
	string internalLoopString1norm="default";
	string internalLoopString2="default";
	string internalLoopStringNormalization="default";
	string internalLoopString1withPartialVectorization="default";
	bool fullvectorization=true;
	bool partialvectorization=true;
	bool ok1=true;
	int loop_count_partialVectorization=0;
	int remainerPartialVectorization=0;
	string initString="default";
	string dotString="default";
	string loop_string_partialVectorization="default";
	string extra_loop_string_partialVectorization="default";
	int vectorSize=0;
    string outputPoolingSelectorString="";
	string endPoolingString="}\n";
	string endPoolingString2="";
	string poolingSelectorString="";

	if (dim.useMaxPooling){
		setPoolingLayerHalf(outputPoolingSelectorString,endPoolingString,endPoolingString2,poolingSelectorString,builder);
	}else{
		builder->set("gresult", "{{gActivationFunction}}");
		setActivationFunctionHalf(builder);
		setNonPoolingLayerVariableHalf(builder,endPoolingString,endPoolingString2,fullvectorization);
	}


	testConditionHalf(ok1);


	int countVectorizationPadding=dim.inputPlanes%4;
	if (countVectorizationPadding!=0)
		fullvectorization=false;


	if (partialvectorization){
		setAutoVectorizationHalf(vectorSize,remainerPartialVectorization,loop_count_partialVectorization,ok1, partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad,initializationCondition,builder,loop_string_partialVectorization, extra_loop_string_partialVectorization, initString, dotString);
	}

	setHintCompilerHalf(batchSize,fullvectorization,partialvectorization,partialVectorizationType,builder);

//	if ((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0)))
//	if (normalization)
//		if (partialvectorization)
//			LOGI("internalLoopStringNormalization");
//		else
//			LOGI("internalLoopString1norm");
//	else
//		if(partialvectorization)
//			LOGI("internalLoopString1withPartialVectorization");
//		else
//			LOGI("internalLoopString1");
//	else
//		LOGI("internalLoopString2");

	setInternalLoopHalf(ok1,loop_count_partialVectorization,internalLoopString1,internalLoopString1norm,internalLoopString2,internalLoopStringNormalization,internalLoopString1withPartialVectorization,initializationCondition,loop_string_partialVectorization,extra_loop_string_partialVectorization,partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad);

	writeKernelcodeHalf(builder, outputPoolingSelectorString,  poolingSelectorString,  partialvectorization,  normalization,  internalLoopStringNormalization,  internalLoopString1norm,  internalLoopString1withPartialVectorization,  internalLoopString1,  internalLoopString2, fullvectorization,  batchSize, ok1);


}

void Forward1::testConditionHalf(bool &ok1){
	if(dim.outputSizeSquared==1){
			for (int u = -dim.halfFilterSize; u <= (dim.halfFilterSize-(dim.filterSize % 2 == 0 ? 1 : 0)); u++){
				int temp = u+dim.halfFilterSize;
				if ((temp < 0 ) || (temp >= dim.inputSize)){
					ok1=false;
					break;
				}
			}
	}else
	ok1=false;
}


void Forward1::setNonPoolingLayerVariableHalf(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization){

	builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
	builder->set("gSumAndBias", dim.biased? "{{gSum}}+bias[(globalId {{DIVgOutputSizeSquared}}) % {{gNumFilters}} ]":"{{gSum}}");
	builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((globalId % {{gOutputSizeSquared}}) % {{gOutputSize}})*"+to_string(dim.stride)+";\n":"int outputCol = (globalId % {{gOutputSizeSquared}}) % {{gOutputSize}};\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = ((globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}})*"+to_string(dim.stride)+";\n":"int outputRow = (globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}};\n":"");
	builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.outputSize):"");
	builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
	builder->set("gSum", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "dot( sum,(half4)(convert_half(1.0f),convert_half(1.0f),convert_half(1.0f),convert_half(1.0f)))":"sum"));
	builder->set("gMaxPooling", "");
	builder->set("gMaxPoolingEnd", "");
}


void  Forward1::setActivationFunctionHalf(TemplatedKernel *builder){


		if (dim.activationLayer==1)
			builder->set("gActivationFunction", "{{gSumAndBias}}");
		if (dim.activationLayer==2)
			builder->set("gActivationFunction", "fmax ( {{gSumAndBias}} , 0 )");
		if (dim.activationLayer==3)
			builder->set("gActivationFunction", "tanh ( {{gSumAndBias}} )");
		if (dim.activationLayer==4)
			builder->set("gActivationFunction", "(1.7159f * tanh(0.66667f * {{gSumAndBias}}))");
		if (dim.activationLayer==5)
			builder->set("gActivationFunction", "(convert_half(1.0f) / (1 + exp(- ({{gSumAndBias}}))))");
		if (dim.activationLayer==6)
			builder->set("gActivationFunction", "fmin (fmax ( {{gSumAndBias}} , 0 ),(exp({{gSumAndBias}}) - 1))");

}


void Forward1::setPoolingLayerHalf(    string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder){
#if TRANSFER ==0
	poolingSelectorString=", global int * selectorArray, global half *gradInput";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionHalf(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="half maxPool=convert_half(-999.99f);\n"
								"    int selectorID=100;\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionHalf(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="half maxPool=convert_half(-999.99f);\n"
									   "    int selectorID=100;\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}
#else
	poolingSelectorString="";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionHalf(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="half maxPool=convert_half(-999.99f);\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionHalf(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="half maxPool=convert_half(-999.99f);\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="\n";
	}
#endif

}

void Forward1::setAutoVectorizationHalf(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString){

		vector<string>indexOpencl;
		indexOpencl.push_back("0");
		indexOpencl.push_back("1");
		indexOpencl.push_back("2");
		indexOpencl.push_back("3");
		indexOpencl.push_back("4");
		indexOpencl.push_back("5");
		indexOpencl.push_back("6");
		indexOpencl.push_back("7");
		indexOpencl.push_back("8");
		indexOpencl.push_back("9");
		indexOpencl.push_back("a");
		indexOpencl.push_back("b");
		indexOpencl.push_back("c");
		indexOpencl.push_back("d");
		indexOpencl.push_back("e");
		indexOpencl.push_back("f");
		int size =dim.filterSize;
		int cpt=0;
		vectorSize=4;
		partialVectorizationType="half4";
		partialVectorizationLoad="(*((__global half4*)&";
		constantMemPartialVectorizationLoad="(*((__constant half4*)&";
		initString="(half4)(convert_half(0.0f),convert_half(0.0f),convert_half(0.0f),convert_half(0.0f))";
		dotString="(half4)(convert_half(1.0f),convert_half(1.0f),convert_half(1.0f),convert_half(1.0f))";

		loop_count_partialVectorization=floor((size)/4);
		remainerPartialVectorization=floor((size)%4);
		if ((not ok1)&&(loop_count_partialVectorization==1)){
			cpt=0;
			initializationCondition="";
			for(int i=-(dim.filterSize >> 1);i<(vectorSize-(dim.filterSize >> 1));i++){
				initializationCondition+="            conditionVector.s"+indexOpencl.at(cpt)+"=((half)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"),0,1)));\n";
				cpt++;
			}
		}


if (loop_count_partialVectorization>=1){

		if ((not ok1)&&(loop_count_partialVectorization!=1)){
			initializationCondition="";

				loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
				loop_string_partialVectorization+="            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(vectorSize);i++){
						loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((half)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}

				if (normalization)
					loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
				else loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

				//loop_string_partialVectorization+= "               sum=dot( inputsV*filterV,"+dotString+");\n            }";
		}else
			if (ok1){

				if ((remainerPartialVectorization)<4){
					int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
					initializationCondition="";
					extra_loop_string_partialVectorization="";
					if (loop_count_partialVectorization>1){
						loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
						loop_string_partialVectorization+="              half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
														 "              half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

						loop_string_partialVectorization+= "              sum+=dot( (inputsV*filterV),"+dotString+");\n            }\n            ";
					}else{

						loop_string_partialVectorization= "            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "sum+=dot( (inputsV*filterV),"+dotString+");\n";

					}

					if (remainerPartialVectorization!=0){

						if ((ok1)&&((dim.filterSize >> 1)==1)){
							extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";

							extra_loop_string_partialVectorization+="             half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																 "             half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(1.0f));\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(0.0f));\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
								else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
							}else{
								extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
/////////////////////////
								if (loop_count_partialVectorization>1){
									extra_loop_string_partialVectorization+="            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}else{
									extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}
/////////////////////////
								//extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
								//									 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(1.0f));\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(0.0f));\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
								else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
							}
						}
				}else{

					initializationCondition="";
					extra_loop_string_partialVectorization="";
					loop_string_partialVectorization= "sum+=dot( (inputsV*filterV),"+dotString+");\n";

					if (remainerPartialVectorization!=0){
						extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
						extra_loop_string_partialVectorization+="          for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
							extra_loop_string_partialVectorization+="            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
															 "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

							cpt=0;
							for(int i=0;i<(remainerPartialVectorization);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(1.0f));\n";
									cpt++;
								}
							for(int i=remainerPartialVectorization;i<(vectorSize);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(0.0f));\n";
									cpt++;
								}

							if (normalization)
								extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
							else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

						}
				}


			}else{

				if (normalization){
					loop_string_partialVectorization= "            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
				}
				else{
					loop_string_partialVectorization= "            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
				}
			}

		if ((not ok1)&&((remainerPartialVectorization)!=0)){


			if ((remainerPartialVectorization)<4){

				int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
				if (loop_count_partialVectorization>1){
					extra_loop_string_partialVectorization="           half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}else{
					extra_loop_string_partialVectorization="            inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
												 "            filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((half)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(0.0f));\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
				else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";

			}else{

				extra_loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
				extra_loop_string_partialVectorization+="            half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((half)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(0.0f));\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
				else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

			}
		}
	}else{

			initializationCondition="";//no need
			loop_string_partialVectorization="";//no need
			int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
			extra_loop_string_partialVectorization="           half4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           half4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
			cpt=0;
			for(int i=0;i<(remainerPartialVectorization);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((half)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
				cpt++;
			}
			for(int i=remainerPartialVectorization;i<(vectorSize);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(convert_half(0.0f));\n";
				cpt++;
			}

			if (normalization)
				extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
			else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";

		}
	}


void Forward1::setHintCompilerHalf(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder){
	int possibleGlobalSize = batchSize * dim.outputCubeSize;
	int possibleWorkgroupsize = std::min(possibleGlobalSize, cl->getMaxWorkgroupSize());

	string hintCompilerString="__attribute__((vec_type_hint(";
	if ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
		hintCompilerString+="half4";
	else{
		if ((not fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
			hintCompilerString+="half";
		else
			if (partialvectorization)
				hintCompilerString+=partialVectorizationType;
			else hintCompilerString+="half";
	}

	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);
}

void Forward1::setInternalLoopHalf(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad){

	internalLoopString1="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += inputs[inputRow + {{inputColIdx}}] * filters[filterRowIdx+v] {{gCondition}};\n"
								"            }\n"
								"        }\n";

	internalLoopString1norm="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += scale*(inputs[inputRow + {{inputColIdx}}]+translate) * filters[filterRowIdx+v] {{gCondition}};\n"
								"            }\n"
								"        }\n";

	internalLoopString2="sum += inputs[inputPlaneIdx] * filters[filterIdx] {{gCondition}};\n";

	internalLoopStringNormalization="";
	if (not ok1)
		internalLoopStringNormalization+=partialVectorizationType+" conditionVector;\n";
	if (loop_count_partialVectorization!=1){

		internalLoopStringNormalization+="#pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}else{
		internalLoopStringNormalization+="#pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}
	internalLoopString1withPartialVectorization="";
	if (not ok1)
		internalLoopString1withPartialVectorization+=partialVectorizationType+" conditionVector;\n";
	if ((ok1)&&((dim.filterSize >> 1)==1)){
		internalLoopString1withPartialVectorization+="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}else{

	internalLoopString1withPartialVectorization+="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+loop_string_partialVectorization+""
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}
}

void Forward1::writeKernelcodeHalf(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString2,bool fullvectorization, int batchSize,bool ok1){

	builder->set("outputPoolingSelector", outputPoolingSelectorString);
	builder->set("gPoolingOutputSelector", poolingSelectorString);

	builder->set("gNormalization", normalization? "    ,\n half translate, half scale":"");
	builder->set("gVectorType",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "half4":"half"));
	builder->set("gVectorTypeInitialization",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "(half4)(convert_half(0.0f),convert_half(0.0f),convert_half(0.0f),convert_half(0.0f))":"convert_half(0.0f)"));


	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gInternalLoop",((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)))? normalization? partialvectorization? internalLoopStringNormalization:internalLoopString1norm:partialvectorization? internalLoopString1withPartialVectorization:internalLoopString1:internalLoopString2);


	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gPlusInputPlaneIdxTimeGInputSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gInputSizeSquared}}");

	builder->set("gBeginFirstLoop",(dim.inputPlanes==1)? "":((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "      #pragma unroll\n    for (int planeId = 0; planeId < "+to_string((dim.inputPlanes/4))+"; planeId++) {\n":"      #pragma unroll\n    for (int planeId = 0; planeId < {{gNumInputPlanes}}; planeId++) {\n"));
	builder->set("gCondition", ok1 ? "":(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0))? "*((half)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputColIdx}},0,1)))":"");
	builder->set("gHalfFilterSizeMinusGEven", (dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0));
	builder->set("gNumInputPlanesTimeGFilterSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? (dim.inputPlanes*dim.filterSizeSquared/4):dim.inputPlanes*dim.filterSizeSquared));
	builder->set("gNumInputPlanesTimeGInputSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? ((dim.inputPlanes*dim.inputSizeSquared)/4):dim.inputPlanes*dim.inputSizeSquared));
	builder->set("gLimit", batchSize * dim.numFilters * dim.outputSize * dim.outputSize);

	builder->set("gBias", dim.biased? ", __constant  half * bias":" ");
	builder->set("gNumExamples", batchSize);
	builder->set("inputRowIdx", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputRowIdx2", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputColIdx",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol + v)" : "(outputCol + v + {{gHalfFilterSize}})": dim.padZeros ? "(v)" : "(v + {{gHalfFilterSize}})");
	builder->set("inputColIdx2",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol)" : "(outputCol + {{gHalfFilterSize}})": dim.padZeros ? "" : "({{gHalfFilterSize}})");
    builder->set("gNumInputPlanes", dim.inputPlanes);
    builder->set("gInputPlanes", dim.inputPlanes);
    builder->set("gInputSize", dim.inputSize);
    builder->set("gInputSizeSquared", dim.inputSizeSquared);
    builder->set("gNumFilters", dim.numFilters);
    builder->set("gFilterSize", dim.filterSize);
    builder->set("gHalfFilterSize",  dim.filterSize >> 1);
    builder->set("gFilterSizeSquared", dim.filterSizeSquared);
    builder->set("gNumOutputPlanes", dim.numFilters);
    builder->set("gOutputPlanes", dim.numFilters);
	builder->set("gOutputSize", dim.outputSize);
    builder->set("gOutputSizeSquared", dim.outputSizeSquared);
    builder->set("gPadZeros", dim.padZeros ? 1 : 0);
    builder->set("gMargin", dim.padZeros ? dim.filterSize >> 1 : 0);
    builder->set("gEven", dim.filterSize % 2 == 0 ? 1 : 0);
    builder->set("gSkip", dim.skip);


}


///////////////////////


///////////////////////


///////////////////////


///////////////////////


///////////////////////


///////////////////////


///////////////////////



void Forward1::setupBuilderConvolveEightBit(TemplatedKernel *builder,int batchSize) {

	string activationFunction("linear");
	string partialVectorizationType="default";
	string partialVectorizationLoad="default";
	string constantMemPartialVectorizationLoad="default";
	string initializationCondition="default";
	string internalLoopString1="default";
	string internalLoopString1norm="default";
	string internalLoopString2="default";
	string internalLoopStringNormalization="default";
	string internalLoopString1withPartialVectorization="default";
	bool fullvectorization=true;
	bool partialvectorization=true;
	bool ok1=true;
	int loop_count_partialVectorization=0;
	int remainerPartialVectorization=0;
	string initString="default";
	string dotString="default";
	string loop_string_partialVectorization="default";
	string extra_loop_string_partialVectorization="default";
	int vectorSize=0;
    string outputPoolingSelectorString="";
	string endPoolingString="}\n";
	string endPoolingString2="";
	string poolingSelectorString="";

	if (dim.useMaxPooling){
		setPoolingLayerEightBit(outputPoolingSelectorString,endPoolingString,endPoolingString2,poolingSelectorString,builder);
	}else{
		builder->set("gresult", "{{gActivationFunction}}");
		builder->set("gresult2", "{{gActivationFunction2}}");
		setActivationFunctionEightBit(builder);
		setNonPoolingLayerVariableEightBit(builder,endPoolingString,endPoolingString2,fullvectorization);
	}


	testConditionEightBit(ok1);


	int countVectorizationPadding=dim.inputPlanes%4;
	if (countVectorizationPadding!=0)
		fullvectorization=false;

	if (partialvectorization){
		setAutoVectorizationEightBit(vectorSize,remainerPartialVectorization,loop_count_partialVectorization,ok1, partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad,initializationCondition,builder,loop_string_partialVectorization, extra_loop_string_partialVectorization, initString, dotString);
	}

	setHintCompilerEightBit(batchSize,fullvectorization,partialvectorization,partialVectorizationType,builder);



	setInternalLoopEightBit(ok1,loop_count_partialVectorization,internalLoopString1,internalLoopString1norm,internalLoopString2,internalLoopStringNormalization,internalLoopString1withPartialVectorization,initializationCondition,loop_string_partialVectorization,extra_loop_string_partialVectorization,partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad);

	writeKernelcodeEightBit(builder, outputPoolingSelectorString,  poolingSelectorString,  partialvectorization,  normalization,  internalLoopStringNormalization,  internalLoopString1norm,  internalLoopString1withPartialVectorization,  internalLoopString1,  internalLoopString2, fullvectorization,  batchSize, ok1);


}
STATIC std::string Forward1::getKernelTemplateConvolveEightBit(LayerDimensions dim) {

	LOGI("i m there0 %d", dim.isLast);
	if ((dim.isLast==0)&&((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)))){
		const char * kernelSource =
						"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
			    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2EightBit(\n"
			    	    "    const __global uchar * restrict inputs, __constant uchar *filters,\n"
			    	    "    global uchar *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
			    	    "    const int globalId = get_global_id(0);\n"
						"{{SelectoBegin}}\n"
			    	    "\n"
			    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
			    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
			    	    "\n"
						"    {{gVectorType}} sum = 0;\n"
						"    {{gMaxPooling}}"
			    	    "        {{gImageRowCoord}}"
			    	    "        {{gImageColCoord}}"
			    	    "        {{gBeginFirstLoop}}\n"
			    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
						"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
			    		"        {{gInternalLoop}}"
			    	    "      {{gEndFirstLoop}}\n"
						"\n"
						"    {{gMaxPoolingEnd}}\n"
			    		"    output[globalId] =convert_uchar(round({{gResultScale}}*({{gresult}}-({{gResultZeroPoint}}))));\n"
						"    {{outputPoolingSelector}}"
			    	    "\n"
						"{{SelectoEnd}}"
			    	    "}\n"
			    	    "\n"
			    	    "";
		LOGI("----------------------------------------------getKernelTemplateConvolveEightBit i m there01");
	    return kernelSource;
	}else{

		LOGI("----------------------------------------------getKernelTemplateConvolveEightBit i m there02");
		#if HALF_ACCURACY==0
		const char * kernelSource =
						"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
						"void kernel {{gHintCompiler}}convolve_imagecubes_float2EightBit(\n"
						"    const __global uchar4 * restrict inputs, __constant  {{gVectorType2}} *filters,\n"
						"    global {{gOutputType}} *output {{gBias2}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
						"    const int globalId = get_global_id(0);\n"
						"{{SelectoBegin}}\n"
						"\n"
						"    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
						"    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
						"\n"
						"    {{gVectorType2}} sum = 0;\n"
						"    {{gMaxPooling}}"
						"        {{gImageRowCoord}}"
						"        {{gImageColCoord}}"
						"        {{gBeginFirstLoop}}\n"
						"            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
						"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
						"        {{gInternalLoop}}"
						"      {{gEndFirstLoop}}\n"
						"\n"
						"    {{gMaxPoolingEnd}}\n"
						"    output[globalId] ={{gresult2}};\n"
						"    {{outputPoolingSelector}}"
						"\n"
						"{{SelectoEnd}}"
						"}\n"
						"\n"
						"";
		#else
				const char * kernelSource =
								"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
								"void kernel {{gHintCompiler}}convolve_imagecubes_float2EightBit(\n"
								"    const __global uchar4 * restrict inputs, __constant  {{gVectorType2}} *filters,\n"
								"    global half *output {{gBias2}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
								"    const int globalId = get_global_id(0);\n"
								"{{SelectoBegin}}\n"
								"\n"
								"    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
								"    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
								"\n"
								"    {{gVectorType2}} sum = 0;\n"
								"    {{gMaxPooling}}"
								"        {{gImageRowCoord}}"
								"        {{gImageColCoord}}"
								"        {{gBeginFirstLoop}}\n"
								"            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
								"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
								"        {{gInternalLoop}}"
								"      {{gEndFirstLoop}}\n"
								"\n"
								"    {{gMaxPoolingEnd}}\n"
								"    output[globalId] ={{gresult2}};\n"
								"    {{outputPoolingSelector}}"
								"\n"
								"{{SelectoEnd}}"
								"}\n"
								"\n"
								"";
		#endif

	    return kernelSource;
	}


/*
#if 1
	const char * kernelSource =
	    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2EightBit(\n"
	    	    "    const __global uchar * restrict inputs, __constant uchar *filters,\n"
	    	    "    global uchar *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
	    	    "    const int globalId = get_global_id(0);\n"
				"{{SelectoBegin}}\n"
	    	    "\n"
	    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
	    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
	    	    "\n"
				"    {{gVectorType}} sum = 0;\n"
				"    {{gMaxPooling}}"
	    	    "        {{gImageRowCoord}}"
	    	    "        {{gImageColCoord}}"
	    	    "        {{gBeginFirstLoop}}\n"
	    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
				"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
	    		"        {{gInternalLoop}}"
	    	    "      {{gEndFirstLoop}}\n"
				"\n"
				"    {{gMaxPoolingEnd}}\n"
	    		"    output[globalId] =convert_uchar(round({{gResultScale}}*({{gresult}}-({{gResultZeroPoint}}))));\n"
				"    {{outputPoolingSelector}}"
	    	    "\n"
				"{{SelectoEnd}}"
	    	    "}\n"
	    	    "\n"
	    	    "";
#else
	const char * kernelSource =
	    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2EightBit(\n"
	    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
	    	    "    global float *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
	    	    "    const int globalId = get_global_id(0);\n"
				"{{SelectoBegin}}\n"
	    	    "\n"
	    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
	    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
	    	    "\n"
				"    {{gVectorType}} sum = 0;\n"
				"    {{gVectorType}} sum2 = 0;\n"
				"    {{gVectorType}} sum3 = 0;\n"
				"    {{gMaxPooling}}"
	    	    "        {{gImageRowCoord}}"
	    	    "        {{gImageColCoord}}"
	    	    "        {{gBeginFirstLoop}}\n"
	    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
				"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
	    		"        {{gInternalLoop}}"
	    	    "      {{gEndFirstLoop}}\n"
				"\n"
				"    {{gMaxPoolingEnd}}\n"
	    		"    output[globalId] = round({{gResultScale}}*({{gresult}}-({{gResultZeroPoint}})));\n"
				"    {{outputPoolingSelector}}"
	    	    "\n"
				"{{SelectoEnd}}"
	    	    "}\n"
	    	    "\n"
	    	    "";
#endif*/


    //return kernelSource;
}



void Forward1::testConditionEightBit(bool &ok1){
	if(dim.outputSizeSquared==1){
			for (int u = -dim.halfFilterSize; u <= (dim.halfFilterSize-(dim.filterSize % 2 == 0 ? 1 : 0)); u++){
				int temp = u+dim.halfFilterSize;
				if ((temp < 0 ) || (temp >= dim.inputSize)){
					ok1=false;
					break;
				}
			}
	}else
	ok1=false;
}


void Forward1::setNonPoolingLayerVariableEightBit(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization){

	builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
	builder->set("gSumAndBias", dim.biased? "{{gSum}}+bias[(globalId {{DIVgOutputSizeSquared}}) % {{gNumFilters}} ]":"{{gSum}}");
	builder->set("gSumAndBias2", dim.biased? "{{gSum2}}+bias[(globalId {{DIVgOutputSizeSquared}}) % {{gNumFilters}} ]":"{{gSum2}}");

	builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((globalId % {{gOutputSizeSquared}}) % {{gOutputSize}})*"+to_string(dim.stride)+";\n":"int outputCol = (globalId % {{gOutputSizeSquared}}) % {{gOutputSize}};\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = ((globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}})*"+to_string(dim.stride)+";\n":"int outputRow = (globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}};\n":"");
	builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.outputSize):"");
	builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
	builder->set("gSum", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "dot( sum,(float4)(1.0f,1.0f,1.0f,1.0f))":"sum"));
	#if HALF_ACCURACY==0
		builder->set("gSum2", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "dot( sum,(float4)(1.0f,1.0f,1.0f,1.0f))":"sum"));
	#else
		builder->set("gSum2", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "dot( sum,(half4)(convert_half(1.0f),convert_half(1.0f),convert_half(1.0f),convert_half(1.0f)))":"sum"));
	#endif
	builder->set("gMaxPooling", "");
	builder->set("gMaxPoolingEnd", "");
}


void  Forward1::setActivationFunctionEightBit(TemplatedKernel *builder){


		if (dim.activationLayer==1)
			builder->set("gActivationFunction", "{{gSumAndBias}}");
		if (dim.activationLayer==2)
			builder->set("gActivationFunction", "fmax ( {{gSumAndBias}} , 0 )");
		if (dim.activationLayer==3)
			builder->set("gActivationFunction", "tanh ( {{gSumAndBias}} )");
		if (dim.activationLayer==4)
			builder->set("gActivationFunction", "(1.7159f * tanh(0.66667f * {{gSumAndBias}}))");
		if (dim.activationLayer==5)
			builder->set("gActivationFunction", "(1.0f / (1 + exp(- ({{gSumAndBias}}))))");
		if (dim.activationLayer==6)
			builder->set("gActivationFunction", "fmin (fmax ( {{gSumAndBias}} , 0 ),(exp({{gSumAndBias}}) - 1))");

		////////////////////

		if (dim.activationLayer==1)
			builder->set("gActivationFunction2", "{{gSumAndBias2}}");
		if (dim.activationLayer==2)
			builder->set("gActivationFunction2", "fmax ( {{gSumAndBias2}} , 0 )");
		if (dim.activationLayer==3)
			builder->set("gActivationFunction2", "tanh ( {{gSumAndBias2}} )");
		if (dim.activationLayer==4)
			builder->set("gActivationFunction2", "(1.7159f * tanh(0.66667f * {{gSumAndBias2}}))");
		if (dim.activationLayer==5)
			builder->set("gActivationFunction2", "(1.0f / (1 + exp(- ({{gSumAndBias2}}))))");
		if (dim.activationLayer==6)
			builder->set("gActivationFunction2", "fmin (fmax ( {{gSumAndBias2}} , 0 ),(exp({{gSumAndBias2}}) - 1))");

}


void Forward1::setPoolingLayerEightBit(    string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder){

#if TRANSFER ==0
	poolingSelectorString=", global int * selectorArray, global float *gradInput";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionEightBit(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="float maxPool=-999.99f;\n"
								"    int selectorID=100;\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionEightBit(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="float maxPool=-999.99f;\n"
									   "    int selectorID=100;\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}
#else
		poolingSelectorString="";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionEightBit(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="float maxPool=-999.99f;\n"
								"    int selectorID=100;\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunctionEightBit(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="float maxPool=-999.99f;\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="\n";
	}
#endif
}
void Forward1::setAutoVectorizationEightBit(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString){

		vector<string>indexOpencl;
		indexOpencl.push_back("0");
		indexOpencl.push_back("1");
		indexOpencl.push_back("2");
		indexOpencl.push_back("3");
		indexOpencl.push_back("4");
		indexOpencl.push_back("5");
		indexOpencl.push_back("6");
		indexOpencl.push_back("7");
		indexOpencl.push_back("8");
		indexOpencl.push_back("9");
		indexOpencl.push_back("a");
		indexOpencl.push_back("b");
		indexOpencl.push_back("c");
		indexOpencl.push_back("d");
		indexOpencl.push_back("e");
		indexOpencl.push_back("f");
		int size =dim.filterSize;
		int cpt=0;
		vectorSize=4;
		partialVectorizationType="float4";
		partialVectorizationLoad="(convert_float4_rtp(*(__global uchar4*)&";//"(*((__global float4*)&";
		constantMemPartialVectorizationLoad="(convert_float4_rtp(*(__constant uchar4*)&";//"(*((__constant float4*)&";
		initString="(float4)(0.0f,0.0f,0.0f,0.0f)";
		dotString="(float4)(1.0f,1.0f,1.0f,1.0f)";

		loop_count_partialVectorization=floor((size)/4);
		remainerPartialVectorization=floor((size)%4);
		if ((not ok1)&&(loop_count_partialVectorization==1)){
			cpt=0;
			initializationCondition="";
			for(int i=-(dim.filterSize >> 1);i<(vectorSize-(dim.filterSize >> 1));i++){
				initializationCondition+="            conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"),0,1)));\n";
				cpt++;
			}
		}


if (loop_count_partialVectorization>=1){

		if ((not ok1)&&(loop_count_partialVectorization!=1)){
			initializationCondition="";

				loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
				loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(vectorSize);i++){
						loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}

				if (normalization)
					loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
				else loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n            }";

				//loop_string_partialVectorization+= "               sum=dot( inputsV*filterV,"+dotString+");\n            }";
		}else
			if (ok1){

				if ((remainerPartialVectorization)<4){
					int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
					initializationCondition="";
					extra_loop_string_partialVectorization="";
					if (loop_count_partialVectorization>1){
						loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
						loop_string_partialVectorization+="              float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
														 "              float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

						loop_string_partialVectorization+= "              sum+=dot((float4)( (({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}}))),"+dotString+");\n            }\n            ";
					}else{

						loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "            sum+=dot( (float4)(({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),"+dotString+");\n";

					}

					if (remainerPartialVectorization!=0){

						if ((ok1)&&((dim.filterSize >> 1)==1)){
							extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";

							extra_loop_string_partialVectorization+="             float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																 "             float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
								else extra_loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
							}else{
								extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
/////////////////////////
								if (loop_count_partialVectorization>1){
									extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}else{
									extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}
/////////////////////////
								//extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
								//									 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
								else extra_loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
							}
						}
				}else{

					initializationCondition="";
					extra_loop_string_partialVectorization="";
					loop_string_partialVectorization= "            sum+=dot( (float4)(({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),"+dotString+");\n";

					if (remainerPartialVectorization!=0){
						extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
						extra_loop_string_partialVectorization+="          for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
							extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
															 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

							cpt=0;
							for(int i=0;i<(remainerPartialVectorization);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
									cpt++;
								}
							for(int i=remainerPartialVectorization;i<(vectorSize);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
									cpt++;
								}

							if (normalization)
								extra_loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n            }";
							else extra_loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n            }";

						}
				}


			}else{

				if (normalization){
					loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
				}
				else{
					loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
				}
			}

		if ((not ok1)&&((remainerPartialVectorization)!=0)){


			if ((remainerPartialVectorization)<4){

				int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
				if (loop_count_partialVectorization>1){
					extra_loop_string_partialVectorization="           float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}else{
					extra_loop_string_partialVectorization="            inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
												 "            filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
				else extra_loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";

			}else{

				extra_loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
				extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n            }";
				else extra_loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n            }";

			}
		}
	}else{

			initializationCondition="";//no need
			loop_string_partialVectorization="";//no need
			int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
			extra_loop_string_partialVectorization="           float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
			cpt=0;
			for(int i=0;i<(remainerPartialVectorization);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
				cpt++;
			}
			for(int i=remainerPartialVectorization;i<(vectorSize);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
				cpt++;
			}

			if (normalization)
				extra_loop_string_partialVectorization+= "            sum+=dot((float4)( scale*({{gInput_scale}}*inputsV+{{gInput_zero_point}}+translate)*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";
			else extra_loop_string_partialVectorization+= "            sum+=dot((float4)( ({{gInput_scale}}*inputsV+{{gInput_zero_point}})*({{gWeight_scale}}*filterV+{{weight_zero_point}})),conditionVector);\n";

		}
	}


void Forward1::setHintCompilerEightBit(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder){
	int possibleGlobalSize = batchSize * dim.outputCubeSize;
	int possibleWorkgroupsize = std::min(possibleGlobalSize, cl->getMaxWorkgroupSize());

	string hintCompilerString="__attribute__((vec_type_hint(";
	if ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
		hintCompilerString+="uchar4";
	else{
		if ((not fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
			hintCompilerString+="uchar";
		else
			if (partialvectorization)
				hintCompilerString+=partialVectorizationType;
			else hintCompilerString+="uchar";
	}

	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);
}

void Forward1::setInternalLoopEightBit(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad){

	internalLoopString1="        #pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
			                    "               sum += (({{gInput_scale}}*(convert_float(inputs[inputRow + {{inputColIdx}}]))+{{gInput_zero_point}})) * ({{gWeight_scale}}*(convert_float(filters[filterRowIdx+v]))+{{weight_zero_point}}) {{gCondition}};\n"
								"            }\n"
								"        }\n";
								//"        }\n";
//								"        sum ={{gInput_scale}}*{{gWeight_scale}}*sum+ {{gInput_zero_point}}*{{weight_zero_point}}*{{gFilterSize}}*{{gFilterSize}}*{{gNumInputPlanes}}+ {{gWeight_scale}}*sum3*{{gInput_zero_point}}+{{gInput_scale}}*sum2*{{weight_zero_point}};\n//";

	internalLoopString1norm="        #pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += scale*(({{gInput_scale}}*inputs[inputRow + {{inputColIdx}}]+{{gInput_zero_point}})+translate) * ({{gWeight_scale}}*filters[filterRowIdx+v]+{{weight_zero_point}}) {{gCondition}};\n"
								"            }\n"
								"        }\n";

#if HALF_ACCURACY==0
	internalLoopString2="sum+=({{gInput_scale}}*(convert_float4_rtp(*(__global uchar4*)&inputs[inputPlaneIdx]))+(float4)({{gInput_zero_point}},{{gInput_zero_point}},{{gInput_zero_point}},{{gInput_zero_point}}))* filters[filterIdx] {{gCondition}};\n";

#else
	internalLoopString2="sum+=(convert_half({{gInput_scale}})*(convert_half4_rtp(*(__global uchar4*)&inputs[inputPlaneIdx]))+(half4)(convert_half({{gInput_zero_point}}),convert_half({{gInput_zero_point}}),convert_half({{gInput_zero_point}}),convert_half({{gInput_zero_point}})))* filters[filterIdx] {{gCondition}};\n";
#endif
	internalLoopStringNormalization="";
	if (not ok1)
		internalLoopStringNormalization+=partialVectorizationType+" conditionVector;\n";
	if (loop_count_partialVectorization!=1){

		internalLoopStringNormalization+="        #pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}else{
		internalLoopStringNormalization+="        #pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}
	internalLoopString1withPartialVectorization="";
	if (not ok1)
		internalLoopString1withPartialVectorization+=partialVectorizationType+" conditionVector;\n";
	if ((ok1)&&((dim.filterSize >> 1)==1)){
		internalLoopString1withPartialVectorization+="        #pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}else{

	internalLoopString1withPartialVectorization+="        #pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+loop_string_partialVectorization+""
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}
}

void Forward1::writeKernelcodeEightBit(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString2,bool fullvectorization, int batchSize,bool ok1){

	builder->set("outputPoolingSelector", outputPoolingSelectorString);
	builder->set("gPoolingOutputSelector", poolingSelectorString);

	builder->set("gNormalization", normalization? "    ,\n float translate, float scale":"");
	builder->set("gVectorType",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "float4":"float"));
	#if HALF_ACCURACY==0
		builder->set("gVectorType2",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "float4":"float"));
	#else
		builder->set("gVectorType2",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "half4":"half"));
	#endif
	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gInternalLoop",((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)))? normalization? partialvectorization? internalLoopStringNormalization:internalLoopString1norm:partialvectorization? internalLoopString1withPartialVectorization:internalLoopString1:internalLoopString2);


	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gPlusInputPlaneIdxTimeGInputSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gInputSizeSquared}}");

	builder->set("gBeginFirstLoop",(dim.inputPlanes==1)? "":((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "      #pragma unroll\n    for (int planeId = 0; planeId < "+to_string((dim.inputPlanes/4))+"; planeId++) {\n":"      #pragma unroll\n    for (int planeId = 0; planeId < {{gNumInputPlanes}}; planeId++) {\n"));
	builder->set("gCondition", ok1 ? "":(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0))? "*((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputColIdx}},0,1)))":"");
	builder->set("gHalfFilterSizeMinusGEven", (dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0));
	builder->set("gNumInputPlanesTimeGFilterSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? (dim.inputPlanes*dim.filterSizeSquared/4):dim.inputPlanes*dim.filterSizeSquared));
	builder->set("gNumInputPlanesTimeGInputSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? ((dim.inputPlanes*dim.inputSizeSquared)/4):dim.inputPlanes*dim.inputSizeSquared));
	builder->set("gLimit", batchSize * dim.numFilters * dim.outputSize * dim.outputSize);

	builder->set("gBias", dim.biased? ", __constant  float * bias":" ");
	#if HALF_ACCURACY==0
		builder->set("gBias2", dim.biased? ", __constant  float * bias":" ");
	#else
		builder->set("gBias2", dim.biased? ", __constant  half * bias":" ");
	#endif
	#if HALF_ACCURACY==0
		builder->set("gOutputType", "float");
	#else
		builder->set("gOutputType", "half");
	#endif
	builder->set("gNumExamples", batchSize);
	builder->set("inputRowIdx", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputRowIdx2", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputColIdx",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol + v)" : "(outputCol + v + {{gHalfFilterSize}})": dim.padZeros ? "(v)" : "(v + {{gHalfFilterSize}})");
	builder->set("inputColIdx2",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol)" : "(outputCol + {{gHalfFilterSize}})": dim.padZeros ? "" : "({{gHalfFilterSize}})");
    builder->set("gNumInputPlanes", dim.inputPlanes);
    builder->set("gInputPlanes", dim.inputPlanes);
    builder->set("gInputSize", dim.inputSize);
    builder->set("gInputSizeSquared", dim.inputSizeSquared);
    builder->set("gNumFilters", dim.numFilters);
    builder->set("gFilterSize", dim.filterSize);
    builder->set("gHalfFilterSize",  dim.filterSize >> 1);
    builder->set("gFilterSizeSquared", dim.filterSizeSquared);
    builder->set("gNumOutputPlanes", dim.numFilters);
    builder->set("gOutputPlanes", dim.numFilters);
	builder->set("gOutputSize", dim.outputSize);
    builder->set("gOutputSizeSquared", dim.outputSizeSquared);
    builder->set("gPadZeros", dim.padZeros ? 1 : 0);
    builder->set("gMargin", dim.padZeros ? dim.filterSize >> 1 : 0);
    builder->set("gEven", dim.filterSize % 2 == 0 ? 1 : 0);
    builder->set("gSkip", dim.skip);
    builder->set("gInput_scale", (1/multI));
    builder->set("gWeight_scale", (1/multW));
    builder->set("gInput_zero_point", (minI));
    builder->set("weight_zero_point", (minW));
    builder->set("gResultScale", (multR));
    builder->set("gResultZeroPoint", (minR));
    LOGI("mult %f min %f",multI, minI);

}
