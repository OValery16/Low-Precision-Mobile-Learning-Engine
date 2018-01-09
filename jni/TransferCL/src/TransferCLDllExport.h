#pragma once

#if defined(_WIN32) 
# if defined(TransferCL_EXPORTS)
#  define TransferCL_EXPORT __declspec(dllexport)
# else
#  define TransferCL_EXPORT __declspec(dllimport)
# endif // TransferCL_EXPORTS
#else // _WIN32
# define TransferCL_EXPORT
#endif


#define PUBLICAPI

#define PUBLIC
#define PROTECTED
#define PRIVATE

typedef unsigned char uchar;

typedef long long int64;
typedef int int32;

#include "dependencies.h"

