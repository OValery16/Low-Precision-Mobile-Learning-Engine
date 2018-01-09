
#include <stdio.h>
#include "transferCLinterface.h"

#include <android/bitmap.h>
#include <jni.h>


#include "trainEngine/train.h"
#include "predictEngine/predict.h"



//#include "kernelManager/kernelManager.h"

#include <sys/stat.h>
#include <dirent.h>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

//#include <Eigen/Dense>
//#include <Eigen/Core>
//#include <Eigen/SVD>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <sstream>

//using Eigen::MatrixXd;
//
//using namespace std;
//using namespace Eigen;

////#include "train.h"
//#define CLMATH_VERBOSE 1
//#define DEEPCL_VERBOSE 1
/*
void  conv2(  MatrixXf I,int rows,int cols , MatrixXf kernel ,int kCols,int kRows,MatrixXf &O)
{
	// find center position of kernel (half of kernel size)
	int kCenterX = kCols / 2;
	int kCenterY = kRows / 2;

	for(int i=0; i < rows; ++i)              // rows
	{
	    for(int j=0; j < cols; ++j)          // columns
	    {
	        for(int m=0; m < kRows; ++m)     // kernel rows
	        {
	            int mm = kRows - 1 - m;      // row index of flipped kernel

	            for(int n=0; n < kCols; ++n) // kernel columns
	            {
	                int nn = kCols - 1 - n;  // column index of flipped kernel

	                // index of input signal, used for checking boundary
	                int ii = i + (m - kCenterY);
	                int jj = j + (n - kCenterX);

	                // ignore input samples which are out of bound
	                if( ii >= 0 && ii < rows && jj >= 0 && jj < cols ){
	                	O(i,j) += I(ii,jj)* kernel(mm,nn);
	                	LOGI("image %d %d",ii,jj);
	                }
	            }
	        }
	        LOGI("---------");
	    }
	}
//	//MatrixXf O=  MatrixXf::Ones(18,18);
//    int kCenterX = 3 / 2;
//    int kCenterY = 3 / 2;
//
//    for(int i=0; i < 18; ++i)              // rows
//    {
//        for(int j=0; j < 18; ++j)          // columns
//        {
//            for(int m=0; m < 3; ++m)     // kernel rows
//            {
//                int mm = 3 - 1 - m;      // row index of flipped kernel
//
//                for(int n=0; n < 3; ++n) // kernel columns
//                {
//                    int nn = 3 - 1 - n;  // column index of flipped kernel
//
//                    // index of input signal, used for checking boundary
//                    int ii = i + (m - kCenterY);
//                    int jj = j + (n - kCenterX);
//
//                    // ignore input samples which are out of bound
//                    if( ii >= 0 && ii < 18 && jj >= 0 && jj < 18 )
//                        O(i,j) += I(ii,jj)* kernel(mm,nn);
//                }
//            }
//        }
//    }
//    std::stringstream buffer2;
//  buffer2 << O << std::endl;
//  LOGI("%s",buffer2.str().c_str());
}



void testEigen(){

	MatrixXf m1= MatrixXf::Ones(18,18);
		  //MatrixXf m2= MatrixXf::Random(3,3);

	MatrixXf m2(3,3);
		  m2 << 0.7176, 0.2996 ,0.0595,
		      0.9916  ,  0.0835  ,  0.8505,
		      0.8343  ,  0.6881  ,  0.0181;

		  MatrixXf m3(18,18);
		  m3.setZero(18,18);
		  //m2(0,0)=0.23;
	//	  Matrix4f m3= Matrix4f::Random(3.0f,3.0f,3.0f);

		  conv2( m1,18,18,m2,3,3,m3);

		  std::stringstream buffer;
		  buffer << m3 << std::endl;
		  LOGI("%s",buffer.str().c_str());
		  LOGI("-------------------------------");

//		  float* o=new float[324];

//		  myConv(m1.data(),18,18,m2.data(),3,3,o);
//		  MatrixXf m3 = Map<MatrixXf>( o, 18, 18 );




		  JacobiSVD<MatrixXf> svd(m2, ComputeThinU | ComputeThinV);

		  MatrixXf U=svd.matrixU() ;
		  MatrixXf S= svd.singularValues().asDiagonal();
		  MatrixXf V= svd.matrixV();
	MatrixXf m4=U.col(0)*S(0,0)*V.transpose().row(0)+U.col(1)*S(1,1)*V.transpose().row(1)+U.col(2)*S(2,2)*V.transpose().row(2);
		  MatrixXf mtemp1=U.col(0)*S(0,0);
		  MatrixXf mtemp2=V.transpose().row(0);
		  MatrixXf mtest1(18,18);
		  mtest1=mtemp1*mtemp2;

		  MatrixXf mtemp3=U.col(1)*S(1,1);
		  MatrixXf mtemp4=V.transpose().row(1);

		  MatrixXf m5(18,18);
		  m5.setZero(18,18);
		  MatrixXf m7(18,18);
		  m7.setZero(18,18);
		  MatrixXf m8(18,18);
		  m8.setZero(18,18);
		  MatrixXf m9(18,18);
		  m9.setZero(18,18);
		  LOGI("------------1,3----------------");
		  conv2( m1,18,18,mtemp3,1,3,m8);
		  LOGI("------------3,1----------------");
		  conv2(m8,18,18,mtemp4,3,1,m9);
		  LOGI("-------------------------------");
		  conv2( m1,18,18,mtemp1,1,3,m5);
		  conv2(m5,18,18,mtemp2,3,1,m7);
		  std::stringstream buffer2;
		  buffer2 << m7 << std::endl;
		  LOGI("%s",buffer2.str().c_str());
		  m7=(m9+m7).eval();;
		  //conv2(m8,18,18,mtemp2,3,1,m7);
//		  MatrixXf m6=conv2( m1,mtemp3);
//		  MatrixXf m7=conv2(m5,mtemp2);//+conv2(m6,mtemp4);
		  Eigen::Tensor<double, 4> W(8,1,3,3);
		  W.setZero();

	//	  W(0,0,1,2) = 1;
	//	  W(0,1,2,0) = 1;
	//	  W(0,2,0,1) = 1;
	//	  W(0,1,0,2) = -1;
	//	  W(0,2,1,0) = -1;
	//	  W(0,0,2,1) = -1;

		  LOGI("-------------------------------");
		  std::stringstream buffer3;
		  buffer3 << m7 << std::endl;
		  LOGI("%s",buffer3.str().c_str());

}*/

char* ReadFile(char *filename)
{
   char *buffer = NULL;
   int string_size, read_size;
   FILE *handler = fopen(filename, "r");
   LOGI( "fopen");

   if (handler)
   {
	   LOGI( "handler");
       // Seek the last byte of the file
       fseek(handler, 0, SEEK_END);
       LOGI( "Seek");
       // Offset from the first to the last byte, or in other words, filesize
       string_size = ftell(handler);
       // go back to the start of the file
       rewind(handler);

       // Allocate a string that can hold it all
       buffer = (char*) malloc(sizeof(char) * (string_size + 1) );

       // Read it all in one operation
       read_size = fread(buffer, sizeof(char), string_size, handler);

       // fread doesn't set it so put a \0 in the last position
       // and buffer is now officially a string
       buffer[string_size] = '\0';

       if (string_size != read_size)
       {
           // Something went wrong, throw away the memory and set
           // the buffer to NULL
           free(buffer);
           buffer = NULL;
       }

       // Always remember to close the file.
       fclose(handler);
    }

    return buffer;
}

int
mkpath(std::string s,mode_t mode)
{
    size_t pre=0,pos;
    std::string dir;
    int mdret;

    if(s[s.size()-1]!='/'){
        // force trailing / so we can handle everything in loop
        s+='/';
    }

    while((pos=s.find_first_of('/',pre))!=std::string::npos){
        dir=s.substr(0,pos++);
        pre=pos;
        if(dir.size()==0) continue; // if leading / first time is 0 length
        if((mdret=mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}

int64_t getTimeNsec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000000000LL + now.tv_nsec;
}
static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    std::vector<char> charsCode;
    const jchar *chars = env->GetStringChars(jStr, NULL);
    jsize len = env->GetStringLength(jStr);
    jsize i;

    for( i=0;i<len; i++){
        int code = (int)chars[i];
        charsCode.push_back( code );
    }

    env->ReleaseStringChars(jStr, chars);
    return std::string(charsCode.begin(), charsCode.end());
}

extern "C" jint
Java_com_transferCL_TransferCLlib_training(JNIEnv* env, jclass clazz, jstring appDirectory, jstring cmdTrain)
{
#if 0
	testEigen();
#else
	 struct timespec start1;
		struct timespec end1;
		double elapsedSeconds1;
		clock_gettime(CLOCK_MONOTONIC, &start1);

		string cmdString2=jstring2string(env,cmdTrain);

		string path="";
		path=path+jstring2string(env,appDirectory);
		string path2=path+"directoryTest/";

		struct timeval start, end;
		/*get the start time*/
		gettimeofday(&start, NULL);


		LOGI("###############");
		for(int i=0;i<1;i++){
		TrainModel* t= new TrainModel(path2);

		#if 0
			string filename_label="/data/data/com.sony.openclexample1/memMapFileLabel60000MNIST.raw";
			string filename_data="/data/data/com.sony.openclexample1/memMapFileData60000MNIST.raw";
			int imageSize=28;
			int numOfChannel=1;//black and white => 1; color =>3
			string storeweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstTransferedHalf.dat";
			string loadweightsfile="";
			string loadnormalizationfile="/data/data/com.sony.openclexample1/preloadingData/normalization.txt";
			string networkDefinition="1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n";
			string validationSet="t10k-images-idx3-ubyte";
			int numepochs=20;
			int batchsize=128;//100;//
			int numtrain=59904;//60000;//
			int numtest=9984;//10000;//
			float learningRate=0.0001f;

			string cmdString="train filename_label="+filename_label;
			cmdString=cmdString+" filename_data="+filename_data;
			cmdString=cmdString+" imageSize="+to_string(imageSize);
			cmdString=cmdString+" numPlanes="+to_string(numOfChannel);
			cmdString=cmdString+" storeweightsfile="+storeweightsfile;
			cmdString=cmdString+" loadweightsfile="+loadweightsfile;
			cmdString=cmdString+" loadnormalizationfile="+loadnormalizationfile;
			cmdString=cmdString+" netdef="+networkDefinition;
			cmdString=cmdString+" numepochs="+to_string(numepochs);
			cmdString=cmdString+" batchsize="+to_string(batchsize);
			cmdString=cmdString+" numtrain="+to_string(numtrain);
			cmdString=cmdString+" numtest="+to_string(numtest);
			cmdString=cmdString+" validatefile="+validationSet;
			cmdString=cmdString+" learningrate="+to_string(learningRate);
			LOGI("I m here");
			t->trainCmd(cmdString,path2);
		}
		#else

		LOGI("cmd %s",cmdString2.c_str());
		t->trainCmd(cmdString2,path2);
//		t->cl->finish();
		delete t;
	}
#endif



	clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds(&end1) - TimeSpecToSeconds(&start1);
	LOGI("3)time %f\n\n",elapsedSeconds1);
	/*get the end time*/
	gettimeofday(&end, NULL);
	/*Print the amount of time taken to execute*/
	LOGI("%f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));

#endif
        return 0;
}



extern "C" jint
Java_com_transferCL_TransferCLlib_prediction(JNIEnv* env, jclass clazz, jstring appDirectory, jstring cmdPrediction)
{

	string path="";
	path=path+jstring2string(env,appDirectory);
	string path2=path+"directoryTest/";
	PredictionModel* p=new PredictionModel(path2);
	string cmdString2=jstring2string(env,cmdPrediction);
	p->predictCmd(cmdString2.c_str());
	delete p;


	return 0;
}

extern "C" jint
Java_com_transferCL_TransferCLlib_prepareFiles(JNIEnv* env, jclass clazz, jstring appDirectory, jstring fileNameStoreData,jstring fileNameStoreLabel, jstring fileNameStoreNormalization, jstring manifestPath, int nbImage, int imagesChannelNb)
{
	string path="";
	path=path+jstring2string(env,appDirectory);
	int mkdirretval;
	//create path
	string stringFileNameStoreData=jstring2string(env,fileNameStoreData);
	string stringFileNameStoreLabel=jstring2string(env,fileNameStoreLabel);
	string stringFileNameStoreNormalization=jstring2string(env,fileNameStoreNormalization);
	string stringManifestPath=jstring2string(env,manifestPath);
	string path2=path+"directoryTest/";
	mkdirretval=mkpath(path2.c_str(),0755);
	LOGI("-------Creation of the working directory");
	if (-1 == mkdirretval)
	{
		LOGI("-----------Directory creation failed: the directory already exists");
	}

	string path3=path+"directoryTest/binariesKernel/";
	mkdirretval=mkpath(path3.c_str(),0755);
	if (-1 == mkdirretval)
	{
		LOGI("-----------Directory creation failed: the directory already exists");
	}
	TrainModel* t= new TrainModel(path2);
	//t->prepareFiles("/sdcard1/character/manifest6.txt",128, 1,"/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw","/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw","/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt");
	t->prepareFiles(stringManifestPath,nbImage, imagesChannelNb,  stringFileNameStoreData, stringFileNameStoreLabel, stringFileNameStoreNormalization);

	delete t;


	return 0;
}

