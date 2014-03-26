// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
 #include <ctype.h>
 #include <unistd.h>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Signatures
char* readSource(const char *sourceFilename); 

unsigned int lS = 5;
int vflag = 0;

int main(int argc, char ** argv)
{
    /////////////////////////////////////
    //               INIT              //
    /////////////////////////////////////
   printf("Running Matrix Inversion program\n\n");

   int index;
   int c;

   opterr = 0;

   while ((c = getopt (argc, argv, "v")) != -1) {
      switch (c)
      {
         case 'v':
            vflag = 1;
            break;
         case '?':
            if (isprint (optopt)) {
               fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            }
            else {
               fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
            return 1;
         default:
            abort ();
      }
   }

   if (optind < argc) {
      lS = atoi(argv[optind]);
   }

   size_t datasize = sizeof(float)*lS*lS;


   float mData[lS * lS];
   float mDataB[lS * lS];
   float lResult[lS * lS];

   for (size_t i=0; i<lS*lS; ++i) {
        mData[i] = rand() / (float)RAND_MAX;
        mDataB[i] = 0;
        lResult[i] = 0;
    }

   
   cl_int status;  // use as return value for most OpenCL functions

   cl_uint numPlatforms = 0;
   cl_platform_id *platforms;
                
   // Query for the number of recongnized platforms
   status = clGetPlatformIDs(0, NULL, &numPlatforms);
   if(status != CL_SUCCESS) {
      printf("clGetPlatformIDs failed\n");
      exit(-1);
   }

   // Make sure some platforms were found 
   if(numPlatforms == 0) {
      printf("No platforms detected.\n");
      exit(-1);
   }

   // Allocate enough space for each platform
   platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
   if(platforms == NULL) {
      perror("malloc");
      exit(-1);
   }

   // Fill in platforms
   clGetPlatformIDs(numPlatforms, platforms, NULL);
   if(status != CL_SUCCESS) {
      printf("clGetPlatformIDs failed\n");
      exit(-1);
   }

   // Print out some basic information about each platform
   printf("%u platforms detected\n", numPlatforms);
   for(unsigned int i = 0; i < numPlatforms; i++) {
      char buf[100];
      printf("Platform %u: \n", i);
      status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                       sizeof(buf), buf, NULL);
      printf("\tVendor: %s\n", buf);
      status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                       sizeof(buf), buf, NULL);
      printf("\tName: %s\n", buf);

      if(status != CL_SUCCESS) {
         printf("clGetPlatformInfo failed\n");
         exit(-1);
      }
   }
   printf("\n");

   cl_uint numDevices = 0;
   cl_device_id *devices;

   // Retrive the number of devices present
   status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, 
                           &numDevices);
   if(status != CL_SUCCESS) {
      printf("clGetDeviceIDs failed\n");
      exit(-1);
   }

   // Make sure some devices were found
   if(numDevices == 0) {
      printf("No devices detected.\n");
      exit(-1);
   }

   // Allocate enough space for each device
   devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
   if(devices == NULL) {
      perror("malloc");
      exit(-1);
   }

   // Fill in devices
   status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices,
                     devices, NULL);
   if(status != CL_SUCCESS) {
      printf("clGetDeviceIDs failed\n");
      exit(-1);
   }   

   // Print out some basic information about each device
   printf("%u devices detected\n", numDevices);
   for(unsigned int i = 0; i < numDevices; i++) {
      char buf[100];
      size_t max_wrkgrp_size;
      printf("Device %u: \n", i);
      status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR,
                       sizeof(buf), buf, NULL);
      printf("\tDevice: %s\n", buf);
      status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
                       sizeof(buf), buf, NULL);
      printf("\tName: %s\n", buf);
      status |= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                               &max_wrkgrp_size, NULL);
      printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n",(int)max_wrkgrp_size);

      if(status != CL_SUCCESS) {
         printf("clGetDeviceInfo failed\n");
         exit(-1);
      }
   }
   printf("\n");

   cl_context context;

   // Create a context and associate it with the devices
   context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
   if(status != CL_SUCCESS || context == NULL) {
      printf("clCreateContext failed\n");
      exit(-1);
   }

   cl_command_queue cmdQueue;

   // Create a command queue and associate it with the device you 
   // want to execute on
   cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
   if(status != CL_SUCCESS || cmdQueue == NULL) {
      printf("clCreateCommandQueue failed\n");
      exit(-1);
   }

   cl_mem d_mData;  // Input buffers on device
   cl_mem d_mDataB;       // Output buffer on device

   // Create a buffer object (d_mData) that contains the data from the host ptr A
   d_mData = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                   datasize, mData, &status);
   if(status != CL_SUCCESS || d_mData == NULL) {
      printf("clCreateBuffer failed\n");
      exit(-1);
   }

   // Create a buffer object (d_mDataB) with enough space to hold the output data
   d_mDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                   datasize, NULL, &status);
   if(status != CL_SUCCESS || d_mDataB == NULL) {
      printf("clCreateBuffer failed\n");
      exit(-1);
   }
   

   cl_program program;
   
   char *source;
   const char *sourceFile = "gaussParallel.cl";
   // This function reads in the source code of the program
   source = readSource(sourceFile);

   //printf("Program source is:\n%s\n", source);

   // Create a program. The 'source' string is the code from the 
   // gaussParallel.cl file.
   program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &status);
   if(status != CL_SUCCESS) {
      printf("clCreateProgramWithSource failed\n");
      exit(-1);
   }

   cl_int buildErr;
   // Build (compile & link) the program for the devices.
   // Save the return value in 'buildErr' (the following 
   // code will print any compilation errors to the screen)
   buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

   // If there are build errors, print them to the screen
   if(buildErr != CL_SUCCESS) {
      printf("Program failed to build.\n");
      cl_build_status buildStatus;
      for(unsigned int i = 0; i < numDevices; i++) {
         clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &buildStatus, NULL);
         if(buildStatus == CL_SUCCESS) {
            continue;
         }

         char *buildLog;
         size_t buildLogSize;
         clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                          0, NULL, &buildLogSize);
         buildLog = (char*)malloc(buildLogSize);
         if(buildLog == NULL) {
            perror("malloc");
            exit(-1);
         }
         clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                          buildLogSize, buildLog, NULL);
         buildLog[buildLogSize-1] = '\0';
         printf("Device %u Build Log:\n%s\n", i, buildLog);   
         free(buildLog);
      }
      exit(0);
   }
   else {
      printf("No build errors\n");
   }


   cl_kernel kernel;

   // Create a kernel from the vector addition function (named "invertParallel")
   kernel = clCreateKernel(program, "invertParallel", &status);
   if(status != CL_SUCCESS) {
      printf("clCreateKernel failed\n");
      exit(-1);
   }

   // Associate the input and output buffers with the kernel 
   status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_mData);
   status |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &lS);
   status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_mDataB);
   if(status != CL_SUCCESS) {
      printf("clSetKernelArg failed\n");
      exit(-1);
   }

   // Define an index space (global work size) of threads for execution.  
   // A workgroup size (local work size) is not required, but can be used.
   size_t globalWorkSize[1];  // There are ELEMENTS threads
   globalWorkSize[0] = 32;//lS;

   // Démarrer le chronomètre
   clock_t start, stop;
   double tm = 0.0;
   start = clock();

   // Execute the kernel.
   // 'globalWorkSize' is the 1D dimension of the work-items
   status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
   if(status != CL_SUCCESS) {
      printf("clEnqueueNDRangeKernel failed\n");
      exit(-1);
   }

   // Read the OpenCL output buffer (d_C) to the host output array (C)
   clEnqueueReadBuffer(cmdQueue, d_mDataB, CL_TRUE, 0, datasize, mDataB, 
                  0, NULL, NULL);

   // Arrêter le chronomètre
   stop = clock();
   tm = (double) (stop-start)/CLOCKS_PER_SEC;

   // Verify correctness
   double sum = 0;

   if (vflag) {
      printf("mData : \n");
      for (int i = 0; i < lS; i++) {
         printf("[");
         for (int k = 0; k < lS; k++) {
            printf("%f, ", mData[i * lS + k]);
         }
         printf("]\n");
      } 
      printf("mDataB : \n");
      for (int i = 0; i < lS; i++) {
         printf("[");
         for (int k = 0; k < lS; k++) {
            printf("%f, ", mDataB[i * lS + k]);
         }
         printf("]\n");
      } 
   }
   for (int i=0;i<lS;i++) {
      for (int j=0;j<lS;j++) {
         for (int k=0;k<lS;k++) {
            lResult[i * lS + j] += mData[i * lS + k] * mDataB[k * lS + j];
         }
         sum += lResult[i * lS + j];
      }
   }
   // printf("lResult : \n");
   // for (int i = 0; i < lS; i++) {
   //    printf("[");
   //    for (int k = 0; k < lS; k++) {
   //       printf("%f, ", lResult[i * lS + k]);
   //    }
   //    printf("]\n");
   // }
   printf("Matrix : %d x %d \n", lS, lS);
   printf("Erreur : %f \n", sum-lS);
   
   // Afficher le temps d'exécution dans le stderr
   printf("Temps d'execution = %f sec\n", tm);

   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseCommandQueue(cmdQueue);
   clReleaseMemObject(d_mData);
   clReleaseMemObject(d_mDataB);
   clReleaseContext(context);

   free(source);
   free(platforms);
   free(devices);

}

char* readSource(const char *sourceFilename) {

   FILE *fp;
   int err;
   int size;

   char *source;

   fp = fopen(sourceFilename, "rb");
   if(fp == NULL) {
      printf("Could not open kernel file: %s\n", sourceFilename);
      exit(-1);
   }
   
   err = fseek(fp, 0, SEEK_END);
   if(err != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }

   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   err = fseek(fp, 0, SEEK_SET);
   if(err != 0) {
      printf("Error seeking to start of file\n");
      exit(-1);
   }

   source = (char*)malloc(size+1);
   if(source == NULL) {
      printf("Error allocating %d bytes for the program source\n", size+1);
      exit(-1);
   }

   err = fread(source, 1, size, fp);
   if(err != size) {
      printf("only read %d bytes\n", err);
      exit(0);
   }

   source[size] = '\0';

   return source;
}