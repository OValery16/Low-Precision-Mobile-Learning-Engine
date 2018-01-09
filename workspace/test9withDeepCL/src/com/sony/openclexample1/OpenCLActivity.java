package com.sony.openclexample1;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DateFormat;
import java.util.Date;

import android.app.Activity;
import android.content.res.AssetManager;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.transferCL.*;

public class OpenCLActivity extends Activity {
	protected static final String TAG = "OpenCLActivity";
	TransferCLlib t;
	
	private static boolean copyAssetFolder(AssetManager assetManager,
            String fromAssetPath, String toPath) {
        try {
            String[] files = assetManager.list(fromAssetPath);
            new File(toPath).mkdirs();
            boolean res = true;
            for (String file : files)
                if (file.contains("."))
                    res &= copyAsset(assetManager, 
                            fromAssetPath + "/" + file,
                            toPath + "/" + file);
                else 
                    res &= copyAssetFolder(assetManager, 
                            fromAssetPath + "/" + file,
                            toPath + "/" + file);
            return res;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean copyAsset(AssetManager assetManager,
            String fromAssetPath, String toPath) {
        InputStream in = null;
        OutputStream out = null;
        try {
          in = assetManager.open(fromAssetPath);
          new File(toPath).createNewFile();
          out = new FileOutputStream(toPath);
          copyFile(in, out);
          in.close();
          in = null;
          out.flush();
          out.close();
          out = null;
          return true;
        } catch(Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    private void makdirectory(){
    	final File directory2 = new File(getDir("execdir",MODE_PRIVATE) + "/../directoryTest/binariesKernel");
    	directory2.mkdirs();
    }

    private static void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
          out.write(buffer, 0, read);
        }
    }
	
	private void copyFile(final String f) {
		InputStream in;
		try {
			in = getAssets().open(f);
			final File of = new File(getDir("execdir",MODE_PRIVATE), f);
			
			final OutputStream out = new FileOutputStream(of);

			final byte b[] = new byte[65535];
			int sz = 0;
			while ((sz = in.read(b)) > 0) {
				out.write(b, 0, sz);
			}
			in.close();
			out.close();
		} catch (IOException e) {       
			e.printStackTrace();
		}
	}
	
	static boolean sfoundLibrary = true;  

	final int info[] = new int[3];

    LinearLayout layout;
    Bitmap bmpOrig, bmpOpenCL, bmpNativeC;
    ImageView imageView;
    TextView textView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        t=new TransferCLlib();    	
  
    }
        
    public void prepareTrainingFiles(View v) {
    	String fileNameStoreData="/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw";
    	String fileNameStoreLabel= "/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw";
    	String fileNameStoreNormalization="/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt";
    	
    	t.prepareFiles("/data/data/com.sony.openclexample1/", fileNameStoreData,fileNameStoreLabel, fileNameStoreNormalization,"/sdcard1/character2/manifest5.txt",150, 1);
    	
//    	//this method prepares the training files (the training file and their labels are respectively stored in one binary file) and the mean And stdDev are stored in one file
//    	String fileNameStoreData="/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw";
//    	String fileNameStoreLabel= "/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw";
//    	String fileNameStoreNormalization="/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt";
//    	
//    	t.prepareFiles("/data/data/com.sony.openclexample1/", fileNameStoreData,fileNameStoreLabel, fileNameStoreNormalization,"/sdcard1/character2/manifest6.txt",128, 1);
    }

    public void trainingModel(View v) {
    	// this method trains our neural network at the native level
    	/*String filename_label="/data/data/com.sony.openclexample1/memMapFileLabel60000MNIST.raw";
    	String filename_data="/data/data/com.sony.openclexample1/memMapFileData60000MNIST.raw";
		int imageSize=224;
		int numOfChannel=3;//black and white => 1; color =>3
		String storeweightsfile="/data/data/com.sony.openclexample1/directoryTest/weightstTransferedTEST.dat";
		String loadweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstTransferedHalf.dat";//"/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//weightstTransferedHalf.dat";//
		String loadnormalizationfile="/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt";
		String networkDefinition="1s64c3z-relu-1s64c3z-mp2-1s128c3z-relu-1s128c3z-mp2-1s256c3z-relu-1s512c3z-mp2-1s512c3z-relu-1s512c3z-mp2-1s512c3z-relu-1s512c3z-mp2-4096n-relu-4096n-relu-2n";// see https://github.com/hughperkins/DeepCL/blob/master/doc/Commandline.md
		int numepochs=200;//2;//
		int batchsize=4;
		int numtrain=4;
		float learningRate=0.01f;   	*/

/*    	String filename_label="/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw";
    	String filename_data="/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw";
		int imageSize=28;
		int numOfChannel=1;//black and white => 1; color =>3
		String storeweightsfile="/data/data/com.sony.openclexample1/directoryTest/weightstTransferedTEST.dat";
		String loadweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//"/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//weightstTransferedHalf.dat";//
		String loadnormalizationfile="/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt";
		String networkDefinition="1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n";// see https://github.com/hughperkins/DeepCL/blob/master/doc/Commandline.md
		int numepochs=100;//2;//1;//
		int batchsize=150;
		int numtrain=150;
		float learningRate=0.004f;*/
    	
    	String filename_label="/data/data/com.sony.openclexample1/memMapFileLabel60000MNIST.raw";
    	String filename_data="/data/data/com.sony.openclexample1/memMapFileData60000MNIST.raw";
		int imageSize=32;
		int numOfChannel=3;//black and white => 1; color =>3
		String storeweightsfile="/data/data/com.sony.openclexample1/directoryTest/weightstTransferedTEST.dat";
		String loadweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//"/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//weightstTransferedHalf.dat";//
		String loadnormalizationfile="/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt";
		String networkDefinition="1s32c5z-mp2-relu-1s32c5z-relu-mp2-1s32c5z-relu-mp2-64n-relu-10n";// see https://github.com/hughperkins/DeepCL/blob/master/doc/Commandline.md
		int numepochs=100;//2;//1;//
		int batchsize=150;
		int numtrain=150;
		float learningRate=0.004f;

		String cmdString="train filename_label="+filename_label;
		cmdString=cmdString+" filename_data="+filename_data;
		cmdString=cmdString+" imageSize="+Integer.toString(imageSize);
		cmdString=cmdString+" numPlanes="+Integer.toString(numOfChannel);
		cmdString=cmdString+" storeweightsfile="+storeweightsfile;
		cmdString=cmdString+" loadweightsfile="+loadweightsfile;
		cmdString=cmdString+" loadnormalizationfile="+loadnormalizationfile;
		cmdString=cmdString+" netdef="+networkDefinition;
		cmdString=cmdString+" numepochs="+Integer.toString(numepochs);
		cmdString=cmdString+" batchsize="+Integer.toString(batchsize);
		cmdString=cmdString+" numtrain="+Integer.toString(numtrain);
		cmdString=cmdString+" learningrate="+Float.toString(learningRate);
    	
		String appDirctory ="/data/data/com.sony.openclexample1/";

    	t.training(appDirctory,cmdString);

    }

    public void predictImages(View v) {
    	// this method performs the prediction and sore the result in a file
    	String appDirctory ="/data/data/com.sony.openclexample1/";
    	String cmdString ="./predict weightsfile=/data/data/com.sony.openclexample1/directoryTest/weightstTransferedTEST.dat  inputfile=/sdcard1/character/manifest4.txt outputfile=/data/data/com.sony.openclexample1/preloadingData/pred2.txt";
    	t.prediction(appDirctory,cmdString);

//    	String appDirctory ="/data/data/com.sony.openclexample1/";
//    	String cmdString ="./predict weightsfile=/data/data/com.sony.openclexample1/directoryTest/weightstTransferedTEST.dat  inputfile=/sdcard1/character2/manifest5.txt outputfile=/data/data/com.sony.openclexample1/preloadingData/pred2.txt";
//    	t.prediction(appDirctory,cmdString);
    }
    
    public static void deleteFiles(String path) {

        File file = new File(path);

        if (file.exists()) {
            String deleteCmd = "rm -r " + path;
            Runtime runtime = Runtime.getRuntime();
            try {
                runtime.exec(deleteCmd);
            } catch (IOException e) { }
        }
    }
    
    
}
