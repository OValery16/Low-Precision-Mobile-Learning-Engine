// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <string>
#include "../EasyCL.h"
#include "LuaTemplater.h"
#include "TemplatedKernel.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC
#define PUBLIC

PUBLIC TemplatedKernel::TemplatedKernel(EasyCL *cl) :
        cl(cl) {
    templater = new LuaTemplater();
}
PUBLIC TemplatedKernel::~TemplatedKernel() {
    delete templater;
}
PUBLIC TemplatedKernel &TemplatedKernel::set(std::string name, int value) {
    templater->set(name, value);
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::set(std::string name, float value) {
    templater->set(name, value);
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::set(std::string name, std::string value) {
    templater->set(name, value);
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::set(std::string name, std::vector< std::string > &value) {
    templater->set(name, value);
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::set(std::string name, std::vector< int > &value) {
    templater->set(name, value);
    return *this;
}
PUBLIC TemplatedKernel &TemplatedKernel::set(std::string name, std::vector< float > &value) {
    templater->set(name, value);
    return *this;
}
PUBLIC CLKernel *TemplatedKernel::buildKernel(std::string uniqueName, std::string filename, std::string templateSource, std::string kernelName) {
	return buildKernel(uniqueName, filename, templateSource, kernelName, true);
}
// we have to include both filename and sourcecode because:
// - we want to 'stirngify' the kernels, so we dont have to copy the .cl files around at dpeloyment, so we need
//   the sourcecode stringified, at build time
// - we want the filename, for lookup purposes, and for debugging messages too
 // do NOT delete the reutrned kernel, or yo uwill get a crash ;-)
PUBLIC CLKernel *TemplatedKernel::buildKernel(std::string uniqueName, std::string filename, std::string templateSource, std::string kernelName, bool useKernelStore) {
//    string instanceName = createInstanceName();
    if(!useKernelStore || !cl->kernelExists(uniqueName)) {
        return _buildKernel(uniqueName, filename, templateSource, kernelName, useKernelStore);
    }
    return cl->getKernel(uniqueName);
}
CLKernel *TemplatedKernel::buildKernelOnline(std::string uniqueName, std::string filename, std::string templateSource, std::string kernelName, bool useKernelStore, std::string t) {
    // cout << "building kernel " << uniqueName << endl;
    string renderedKernel = templater->render(templateSource);
//    LOGI("buildKernelOnline");
//    int index = renderedKernel.find("remplacer", 0);
//    renderedKernel.replace(index,9,t);
//LOGI("buildKernelOnline2");
    std::ofstream out;
    string s=cl->absolutePath+"kernelcode2.txt";
       	     out.open(s.c_str()/*"/data/data/com.sony.openclexample1/preloadingData/kernelcode.txt"*/, std::ios::app);
    	     out << templateSource;
    	     out << "\n";
    	     out << renderedKernel;
    	     out.close();
    LOGI("begin");
	//CLKernel *kernel = cl->buildKernelFromStringOnlineBuilding("testNoId",renderedKernel, kernelName, "", "forward.cl");
	CLKernel *kernel = cl->buildKernelFromStringOnlineBuilding(uniqueName,renderedKernel, kernelName, "", filename);
			/*(renderedKernel, kernelName, "", "forwad.cl", false);*/
//			buildKernelFromString(uniqueName,renderedKernel, kernelName, "", filename);

	return kernel;
}

CLKernel *TemplatedKernel::_buildKernel(std::string uniqueName, std::string filename, std::string templateSource, std::string kernelName, bool useKernelStore) {
    // cout << "building kernel " << uniqueName << endl;
    string renderedKernel = templater->render(templateSource);

    // cout << "renderedKernel=" << renderedKernel << endl;
//    CLKernel *kernel = cl->buildKernelFromString(renderedKernel, kernelName, "", filename);

    std::ofstream out;
    string s=cl->absolutePath+"kernelcode.txt";
       	     out.open(s.c_str()/*"/data/data/com.sony.openclexample1/preloadingData/kernelcode.txt"*/, std::ios::app);
    	     out << uniqueName;
    	     out << "\n";
    	     out << renderedKernel;
    	     out.close();

	if(useKernelStore) {
//
	////olivier
	    std::ofstream out;
	     out.open("/data/data/com.sony.openclexample1/preloadingData/forwardcode.txt", std::ios::app);
	     out << renderedKernel;
	     out.close();
	    //cl->storeKernel(uniqueName, kernel, true);
	}
	CLKernel *kernel = cl->buildKernelFromString(uniqueName,renderedKernel, kernelName, "", filename);

	return kernel;
}

CLKernel *TemplatedKernel::buildKernelTest(std::string uniqueName, std::string filename, std::string templateSource, std::string kernelName, bool useKernelStore, std::string t) {
    // cout << "building kernel " << uniqueName << endl;
    string renderedKernel = templater->render(templateSource);
    int index = renderedKernel.find("remplacer", 0);
    renderedKernel.replace(index,9,t);

    // cout << "renderedKernel=" << renderedKernel << endl;
//    CLKernel *kernel = cl->buildKernelFromString(renderedKernel, kernelName, "", filename);

    std::ofstream out;
    string s=cl->absolutePath+"kernelcode.txt";
       	     out.open(s.c_str()/*"/data/data/com.sony.openclexample1/preloadingData/kernelcode.txt"*/, std::ios::app);
    	     out << uniqueName;
    	     out << "\n";
    	     out << renderedKernel;
    	     out.close();

	if(useKernelStore) {
//
	////olivier
	    std::ofstream out;
	     out.open("/data/data/com.sony.openclexample1/preloadingData/forwardcode.txt", std::ios::app);
	     out << renderedKernel;
	     out.close();
	    //cl->storeKernel(uniqueName, kernel, true);
	}
	CLKernel *kernel = cl->buildKernelFromString(uniqueName,renderedKernel, kernelName, "", filename);

	return kernel;
}


// this is mostly for debugging purposes really
PUBLIC std::string TemplatedKernel::getRenderedKernel(std::string templateSource) {
    return templater->render(templateSource);
}

