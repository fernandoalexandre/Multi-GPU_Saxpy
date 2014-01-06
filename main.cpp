/*
 * main.cpp
 *
 *  Created on: 12 de November de 2013
 *      Author: Fernando Alexandre
 *
 *  This program is an example of a multi-GPU computation (within a single system),
 *  that applies a Saxpy matrix operation decomposed evenly among the GPUs.
 *  Additionally it is possible to further decompose the data-sets so each GPU
 *  receives multiple, parallel executions (overlapping partitions).
 *
 *
 *
 *   The MIT License (MIT)
 *
 *   Copyright (c) 2014 Fernando Alexandre
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *   THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

// Default number of overlapping partitions within each GPU
#define DEFAULT_OVERLAP 2
// The alpha value used by Saxpy
#define ALPHA_SCALAR 5.0

// Location of the OpenCL computation
const std::string kernelFile = "saxpy.cl";


// Input/Output data-sets
float *inValues1, *inValues2, *outValues;

// Input/Output memory locations on the GPUs
// GPU Index -> Overlap Index -> cl_mem pointer
cl_mem ** input1, ** input2, ** output;

cl_context ** contexts;

// OpenCL Queues used to trigger data-transfers and computations
// GPU Index -> Overlap Index -> queue pointer
cl_command_queue ** commandQueues;

cl_program ** saxpyProgram;
cl_kernel ** saxpyKernel;

// Number of overlapping partitions within each GPU
unsigned int numOverlap;
// Number of elements in the matrix
unsigned int numberElems;
// Number of devices to be used
unsigned int numDevices;

// Work size used by OpenCL for each OpenCL execution
// (currently all have the same size because of even partitioning)
size_t globalWorkSize[1];

const char* errorString(int error);

// This auxiliary function has been adapted from work done by Ricardo Marques
// in the context of his thesis.
cl_program programFromSource(const std::string &fileName, const cl_context context){
    std::ifstream kernelFile(fileName.data(), std::ios::in);
    if (!kernelFile.is_open()){
        kernelFile.close();
        throw;
    }
    int errcode;
    cl_program program;
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &errcode);
    if(errcode != CL_SUCCESS){
        std::cerr << errorString(errcode) << std::endl;
        return NULL;
    }
    return program;
}


// Initialization function for the OpenCL platform and all OpenCL resources
// required by this program.
void initOpenCL() {
    int errcode = 0;
    unsigned int i, j;
    cl_platform_id platformId;
    numDevices = 0;
    cl_device_id deviceIds[8];

    // Initialize the OpenCL platform itself
    clGetPlatformIDs(1, &platformId, NULL);
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 8, &deviceIds[0], &numDevices);

    contexts = (cl_context **) calloc(sizeof(cl_context *), numDevices);
    commandQueues = (cl_command_queue **) calloc(sizeof(cl_command_queue*), numDevices);
    saxpyProgram = (cl_program **) calloc(sizeof(cl_program *), numDevices);

    globalWorkSize[0] = numberElems / (numDevices * numOverlap);


    input1 = (cl_mem **) calloc(sizeof(cl_mem *), numDevices);
    input2 = (cl_mem **) calloc(sizeof(cl_mem *), numDevices);
    output = (cl_mem **) calloc(sizeof(cl_mem *), numDevices);

    saxpyKernel = (cl_kernel **) calloc(sizeof(cl_kernel *), numDevices);

    // Create the resources for each GPU
    for(i = 0; i < numDevices; i++) {
        contexts[i] = (cl_context *) calloc(sizeof(cl_context), numOverlap);
        commandQueues[i] = (cl_command_queue *) calloc(sizeof(cl_command_queue), numOverlap);

        saxpyProgram[i] = (cl_program *) calloc(sizeof(cl_program), numOverlap);

        input1[i] = (cl_mem *) calloc(sizeof(cl_mem), numOverlap);
        input2[i] = (cl_mem *) calloc(sizeof(cl_mem), numOverlap);
        output[i] = (cl_mem *) calloc(sizeof(cl_mem), numOverlap);

        saxpyKernel[i] = (cl_kernel *) calloc(sizeof(cl_kernel), numDevices);

        // Create resources within each GPU
        for(j = 0; j < numOverlap; j++) {
            contexts[i][j] = clCreateContext(0, 1, &deviceIds[i], NULL, NULL, &errcode);

            // Create a command queue so that the host can issue orders to the device
            commandQueues[i][j] = clCreateCommandQueue(contexts[i][j], deviceIds[i], 0, &errcode);
            // Create buffer objects for X matrix
            input1[i][j] = clCreateBuffer(contexts[i][j], CL_MEM_READ_WRITE, sizeof(cl_float) * globalWorkSize[0], NULL, &errcode);
            if(errcode != 0) {
                std::cerr << "CreateBuffer (X): " << errorString(errcode) << std::endl;
            }
            // Create buffer objects for Y Matrix
            input2[i][j] = clCreateBuffer(contexts[i][j], CL_MEM_READ_WRITE, sizeof(cl_float) * globalWorkSize[0], NULL, &errcode);
            if(errcode != 0) {
                std::cerr << "CreateBuffer (Y): " << errorString(errcode) << std::endl;
            }
            // Create buffer objects for the output
            output[i][j] = clCreateBuffer(contexts[i][j], CL_MEM_READ_WRITE, sizeof(cl_float) * globalWorkSize[0], NULL, &errcode);
            if(errcode != 0) {
                std::cerr << "CreateBuffer (output): " << errorString(errcode) << std::endl;
            }

            // Load and build OpenCL kernels
            saxpyProgram[i][j] = programFromSource(kernelFile, contexts[i][j]);
            errcode = clBuildProgram(saxpyProgram[i][j], 0, NULL, NULL, NULL, NULL);
            if(errcode != 0) {
                char buildlog[16000];
                clGetProgramBuildInfo(saxpyProgram[i][j], deviceIds[i], CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
                std::cerr << "Error in clBuildProgram " << buildlog << std::endl;
                exit(1);
            }
            // Create the kernel object
            saxpyKernel[i][j] = clCreateKernel(saxpyProgram[i][j], "saxpy", &errcode);
            if(errcode != 0) {
                std::cerr << "Error creating kernel: " << errorString(errcode) << std::endl;
            }
        }
    }
}

// Cleans up all the OpenCL resources.
void finishOpenCL() {
    for(unsigned int i = 0; i < numDevices; i++) {
        for(unsigned int j = 0; j < numOverlap; j++) {
            clReleaseMemObject(input1[i][j]);
            clReleaseMemObject(input2[i][j]);
            clReleaseMemObject(output[i][j]);
            clReleaseKernel(saxpyKernel[i][j]);
            clReleaseCommandQueue(commandQueues[i][j]);
            clReleaseContext(contexts[i][j]);
            clReleaseProgram(saxpyProgram[i][j]);
        }
    }
}

// Launch a computation in a devices, overlap partition index and number of this partition
void run(unsigned int device, unsigned int overlap, unsigned int numPartition) {
    int errcode = 0;
    float alpha = ALPHA_SCALAR;

    unsigned int start = numPartition * (sizeof(cl_float) * globalWorkSize[0]);

    // Write input buffers to device memory
    errcode = clEnqueueWriteBuffer(commandQueues[device][overlap], input1[device][overlap], CL_FALSE, 0, sizeof(cl_float) * globalWorkSize[0], inValues1 + start, 0, NULL, NULL);
    if(errcode != 0) {
        std::cerr << "WriteBuffer (X): " << errorString(errcode) << std::endl;
    }

    errcode = clEnqueueWriteBuffer(commandQueues[device][overlap], input2[device][overlap], CL_FALSE, 0, sizeof(cl_float) * globalWorkSize[0], inValues2 + start, 0, NULL, NULL);
    if(errcode != 0) {
        std::cerr << "WriteBuffer (Y): " << errorString(errcode) << std::endl;
    }

    // Set the Saxpy kernel arguments
    errcode = clSetKernelArg(saxpyKernel[device][overlap], 0, sizeof(cl_mem), (void*) &input1[device][overlap]);
    if(errcode != 0) {
        std::cerr << "SetArg0 (X): " << errorString(errcode) << std::endl;
    }
    errcode = clSetKernelArg(saxpyKernel[device][overlap], 1, sizeof(cl_mem), (void*) &input2[device][overlap]);
    if(errcode != 0) {
        std::cerr << "SetArg1 (Y): " << errorString(errcode) << std::endl;
    }
    errcode = clSetKernelArg(saxpyKernel[device][overlap], 2, sizeof(float), (void*) &alpha);
    if(errcode != 0) {
        std::cerr << "SetArg2 (alpha): " << errorString(errcode) << std::endl;
    }
    errcode = clSetKernelArg(saxpyKernel[device][overlap], 3, sizeof(cl_mem), (void*) &output[device][overlap]);
    if(errcode != 0) {
        std::cerr << "SetArg3 (output): " << errorString(errcode) << std::endl;
    }

    // Queue the Saxpy kernel for execution
    errcode = clEnqueueNDRangeKernel(commandQueues[device][overlap], saxpyKernel[device][overlap], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if(errcode != 0) {
        std::cerr << "NDRange: " << errorString(errcode) << std::endl;
    }
    errcode = clEnqueueReadBuffer(commandQueues[device][overlap], output[device][overlap], CL_FALSE, 0, sizeof(cl_float) * globalWorkSize[0], outValues + start, 0, NULL, NULL);
    if(errcode != 0) {
        std::cerr << "ReadBuffer: " << errorString(errcode) << std::endl;
    }
}

void startExecution() {
    unsigned int i, j, acc = 0;

    // Launch all computations (they are non-blocking)
    // Ideally these should be threaded.
    for(i = 0; i < numDevices; i++) {
        for(j = 0; j < numOverlap; j++) {
            run(i, j, acc);
            acc++;
        }
    }

    // Wait for all computations to finish
    for(i = 0; i < numDevices; i++) {
        for(j = 0; j < numOverlap; j++) {
            clFinish(commandQueues[i][j]);
        }
    }
}

int main(int argc, char const *argv[])
{
    unsigned int i;
    if(argc < 2) {
        std::cout << "Usage: " << argv[0] << "<numberElements> <Optional_numberOverlapPerGPU>" << std::endl;
        return -1;
    } else {
        // Set the values in regards to the supplied arguments
        if(argc >= 2) {
            numberElems = atoi(argv[1]);
            if(argc == 3) {
                numOverlap = atoi(argv[2]);
            } else {
                numOverlap = DEFAULT_OVERLAP;
            }
        }
    }

    inValues1 = new float[numberElems];
    inValues2 = new float[numberElems];
    outValues = new float[numberElems];

    // Initialize the example matrices
    for(i = 0; i < numberElems; i++) {
        inValues1[i] = 10;
        inValues2[i] = 15;
    }

    initOpenCL();

    startExecution();

    finishOpenCL();

    free(inValues1);
    free(inValues2);
    free(outValues);
    return 0;
}

// Function adapted from the work of Ricardo Marques in the context
// of his thesis.
// Translates an OpenCL error code to a string.
const char* errorString(int error) {
    switch (error) {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "Invalid global work size";
    default: return "Unknown error";
    }
}
