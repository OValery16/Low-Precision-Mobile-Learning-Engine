// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>

#include "EasyCL.h"
#include "CLWrapper.h"

// this wraps an existing array, which we wont free, rather than creating a new array
// probably more efficient....
class CLCharWrapper : public CLWrapper {
protected:
    unsigned char *hostarray;  // NOT owned by this object, do NOT free :-)
public:
    CLCharWrapper(int N, unsigned char *_hostarray, EasyCL *easycl) :
             CLWrapper(N, easycl),
             hostarray(_hostarray)
              {
    }
    CLCharWrapper(const CLCharWrapper &source) :
        CLWrapper(0, 0), hostarray(0) { // copy constructor
        throw std::runtime_error("can't assign these...");
    }
    CLCharWrapper &operator=(const CLCharWrapper &two) { // assignment operator
       if(this == &two) { // self-assignment
          return *this;
       }
       throw std::runtime_error("can't assign these...");
    }
    inline unsigned char get(int index) {
        return hostarray[index];
    }
    virtual ~CLCharWrapper() {
    }
    virtual int getElementSize() {
    	LOGI("getElementSize0 %d",sizeof(hostarray[0]));
        return sizeof(hostarray[0]);
    }
    virtual void *getHostArray() {
        return hostarray;
    }
    virtual void const*getHostArrayConst() {
        return hostarray;
    }
};

