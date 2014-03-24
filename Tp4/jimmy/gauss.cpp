// System includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

// OpenCL includes
#include <OpenCL/opencl.h>

// Signatures
char* readSource(const char *sourceFilename);
inline void checkErr(cl_int err, const char * texte);
float trouverMax(float *tableauDeFloat, unsigned int lS, cl_context p_context, cl_program p_program, cl_command_queue p_cmdQueue);
void initOpenCL(cl_platform_id *p_platforms, cl_device_id *p_devices, cl_context *p_context, cl_command_queue *p_cmdQueue);
void compileProgram(cl_device_id *p_devices, cl_context p_context, cl_program *p_programme);

int main(int argc, char ** argv)
{
    printf("Running Matrix Inversion pro®gram\n\n");
    
    srand((unsigned int)time(NULL));
    
    unsigned int lS = 100;
    std::cout << "Taille du tableau : " << lS << std::endl;
    //size_t datasize = sizeof(float) * lS * lS;
    
    //float matriceEntree[lS * lS];
    //float matriceSortie[lS * lS];
    /*
    for (size_t i = 0; i < lS * lS ; ++i) {
        //matriceEntree[i] = rand() / (float)RAND_MAX;
        matriceEntree[i] = (int)(rand() % 10) + 1;
        
        matriceSortie[i] = 0;
    }*/
    
    // Initialisation des variables OpenCL
    cl_int cl_status;  // use as return value for most OpenCL functions
    
    cl_platform_id *cl_platforms = NULL;
    cl_device_id *cl_devices = NULL;
    cl_context cl_context;
    cl_command_queue cl_cmdQueue;
    initOpenCL(cl_platforms, cl_devices, &cl_context, &cl_cmdQueue);
    
    
    /////////////////////////////////////
    //            PROGRAM              //
    /////////////////////////////////////
    cl_program cl_program;
    compileProgram(cl_devices, cl_context, &cl_program);
    
    float *matriceReduce = new float[lS];
    
    for (unsigned long long i = 0; i < lS ; ++i) {
        matriceReduce[i] = rand() / (float)RAND_MAX;
        //matriceReduce[i] = (int)(rand() % lS) + 1;
        
    }

    std::cout << trouverMax(matriceReduce, lS, cl_context, cl_program, cl_cmdQueue);
    std::cout << std::endl;
    
    std::cout << trouverMax(matriceReduce, lS, cl_context, cl_program, cl_cmdQueue);
    std::cout << std::endl;
    
    trouverMax(NULL, 0, NULL, NULL, NULL);
    
    /////////////////////////////////////
    //            BUFFERS              //
    /////////////////////////////////////
    // Tableaux pour contenir les données à réduire.
    /*
     
    cl_mem cl_matriceReduce;
    cl_matriceReduce = clCreateBuffer(cl_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, lS * sizeof(float), matriceReduce, &cl_status);
    checkErr(cl_status, "Impossible de créer le buffer d'entrée de test pour la matrice à l'aide de clCreateBuffer");
    
    cl_mem cl_matriceSortieReduce;
    cl_matriceSortieReduce = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, lS * sizeof(float), NULL, &cl_status);
    checkErr(cl_status, "Impossible de créer le buffer de sortie de test pour la matrice à l'aide de clCreateBuffer");
    
    /////////////////////////////////////
    //          KERNEL REDUCE          //
    /////////////////////////////////////
    cl_kernel cl_kernelReduce;
    
    // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
    cl_kernelReduce = clCreateKernel(cl_program, "reduce", &cl_status);
    checkErr(cl_status, "Impossible de créer le kernel cl_kernelReduce à partir du programme à l'aide de clCreateKernel");
    
    // Associate the input and output buffers with the kernel
    cl_status  = clSetKernelArg(cl_kernelReduce, 0, sizeof(cl_mem), &cl_matriceReduce);
    checkErr(cl_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    cl_status |= clSetKernelArg(cl_kernelReduce, 1, 32 * 1024, NULL);
    checkErr(cl_status, "Impossible de créer l'argument 1 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    cl_status |= clSetKernelArg(cl_kernelReduce, 2, sizeof(unsigned int), &lS);
    checkErr(cl_status, "Impossible de créer l'argument 2 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    cl_status |= clSetKernelArg(cl_kernelReduce, 3, sizeof(cl_mem), &cl_matriceSortieReduce);
    checkErr(cl_status, "Impossible de créer l'argument 3 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    */
    /////////////////////////////////////
    //          EXECUTION              //
    /////////////////////////////////////
    // Le nombre d'éléments du tableau
    
    /////////////////////////////////////
    //          VALIDATION             //
    /////////////////////////////////////
    /*
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
    */
    /////////////////////////////////////
    //            CLEANUP              //
    /////////////////////////////////////
    /*
    clReleaseKernel(cl_kernelReduce);
    clReleaseProgram(cl_program);
    clReleaseCommandQueue(cl_cmdQueue);
    clReleaseMemObject(cl_matriceReduce);
    clReleaseMemObject(cl_matriceSortieReduce);
    clReleaseContext(cl_context);
    
    
    free(cl_devices);
    free(cl_platforms);
    */
    
}

/////////////////////////////////////
//         SOURCE READER           //
/////////////////////////////////////
char* readSource(const char *sourceFilename) {
    
    FILE *fp;
    size_t err;
    long size;
    size_t byte_to_read = 1;
    
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
        printf("Error allocating %ld bytes for the program source\n", size+1);
        exit(-1);
    }
    
    err = fread(source, byte_to_read, size, fp);
    if(err != size) {
        printf("only read %zu bytes\n", err);
        exit(0);
    }
    
    source[size] = '\0';
    
    return source;
}

inline void checkErr(cl_int err, const char * texte)
{
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur : " << texte
        << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void initOpenCL(cl_platform_id *p_platforms, cl_device_id *p_devices, cl_context *p_context, cl_command_queue *p_cmdQueue) {
    cl_int l_status;
    cl_uint l_numPlatforms;
    
    cl_uint l_numDevices = 0;
    
    /////////////////////////////////////
    //          PLATEFORM              //
    /////////////////////////////////////
    
    l_status = clGetPlatformIDs(0, NULL, &l_numPlatforms);
    checkErr(l_status, "Impossible d'obtenir le nombre de plateformes à l'aide de clGetPlatformIDs");
    
    // On s'assure qu'au moins une plateforme à été retrouvé.
    if (l_numPlatforms == 0) {
        std::cerr << "Aucune plateforme trouvée." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Allouer de la mémoire pour toute les plateformes
    p_platforms = (cl_platform_id*)malloc(l_numPlatforms * sizeof(cl_platform_id));
    if (p_platforms == NULL) {
        std::cerr << "Impossible d'allouer l'espace pour toute les plateformes." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Obtention des plateformes
    l_status = clGetPlatformIDs(l_numPlatforms, p_platforms, NULL);
    checkErr(l_status, "Impossible d'obtenir les plateformes à l'aide de clGetPlatformIDs");
    
    // Affichage des informations de base de la plateforme
    std::cout << l_numPlatforms << " plateformes trouvée." << std::endl;
    for (unsigned int i = 0; i < l_numPlatforms; i++) {
        char buf[100];
        std::cout << "Plateforme " << i << std::endl;
        
        l_status = clGetPlatformInfo(p_platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
        checkErr(l_status, "Impossible d'obtenir le vendeur de la plateforme à l'aide de clGetPlatformInfo");
        std::cout << "\tVendor : " << buf << std::endl;
        
        l_status = clGetPlatformInfo(p_platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        checkErr(l_status, "Impossible d'obtenir le nom de la plateforme à l'aide de clGetPlatformInfo");
        std::cout << "\tName : " << buf << std::endl;
        
    }
    
    std::cout << std::endl;
    
    /////////////////////////////////////
    //             DEVICE              //
    /////////////////////////////////////
    
    // Obtenir le nombre de devices
    l_status = clGetDeviceIDs(p_platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &l_numDevices);
    checkErr(l_status, "Impossible d'obtenir le nombre de devices à l'aide de clGetDeviceIDs");
    
    // Make sure some devices were found
    if (l_numDevices == 0) {
        std::cerr << "Aucune devices retrouvé." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Allouer de la mémoire pour toute les devices
    p_devices = (cl_device_id*)malloc(l_numDevices * sizeof(cl_device_id));
    if (p_devices == NULL) {
        std::cerr << "Impossible d'allouer l'espace pour toute les devices." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Obtenir les devices
    l_status = clGetDeviceIDs(p_platforms[0], CL_DEVICE_TYPE_GPU, l_numDevices, p_devices, NULL);
    checkErr(l_status, "Impossible d'obtenir les devices à l'aide de clGetDeviceIDs");
    
    // Afficher des informations de base à propos des devices
    std::cout << l_numDevices << " devices trouvé" << std::endl;
    for (unsigned int i = 0; i < l_numDevices; i++) {
        char buf[100];
        std::cout << "Device " << i << std::endl;
        l_status = clGetDeviceInfo(p_devices[i], CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
        std::cout << "\tDevice : " << buf << std::endl;
        checkErr(l_status, "Impossible d'obtenir le vendeur de la device à l'aide de clGetDeviceInfo");
        
        l_status = clGetDeviceInfo(p_devices[i], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
        checkErr(l_status, "Impossible d'obtenir le nom de la device à l'aide de clGetDeviceInfo");
        std::cout << "\tName : " << buf << std::endl;
        
        char l_deviceName[100];
        char l_deviceVendor[100];
        char l_driverVersion[100];
        char l_deviceVersion[100];
        char l_deviceType[100];
        cl_uint l_deviceMaxComputeUnits;
        cl_uint l_deviceMaxWorkGroupSize;
        char l_deviceSingleFpConfig[100];
        
        clGetDeviceInfo(p_devices[i], CL_DEVICE_NAME, sizeof(l_deviceName), l_deviceName, NULL);
        clGetDeviceInfo(p_devices[i], CL_DEVICE_VENDOR, sizeof(l_deviceVendor), l_deviceVendor, NULL);
        clGetDeviceInfo(p_devices[i], CL_DRIVER_VERSION, sizeof(l_driverVersion), l_driverVersion, NULL);
        clGetDeviceInfo(p_devices[i], CL_DEVICE_VERSION, sizeof(l_deviceVersion), l_deviceVersion, NULL);
        clGetDeviceInfo(p_devices[i], CL_DEVICE_TYPE, sizeof(l_deviceType), l_deviceType, NULL);
        clGetDeviceInfo(p_devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(l_deviceMaxComputeUnits), &l_deviceMaxComputeUnits, NULL);
        clGetDeviceInfo(p_devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(l_deviceMaxWorkGroupSize), &l_deviceMaxWorkGroupSize, NULL);
        clGetDeviceInfo(p_devices[i], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(l_deviceSingleFpConfig), &l_deviceSingleFpConfig, NULL);
    
        std::cout << "\tCL_DEVICE_NAME:                        " << l_deviceName << std::endl;
        std::cout << "\tCL_DEVICE_VENDOR:                      " << l_deviceVendor << std::endl;
        std::cout << "\tCL_DRIVER_VERSION:                     " << l_driverVersion << std::endl;
        std::cout << "\tCL_DEVICE_VERSION:                     " << l_deviceVersion << std::endl;
        std::cout << "\tCL_DEVICE_MAX_COMPUTE_UNITS:           " << l_deviceMaxComputeUnits << std::endl;
        std::cout << "\tCL_DEVICE_MAX_WORK_GROUP_SIZE:         " << l_deviceMaxWorkGroupSize << std::endl;
        std::cout << "\tCL_DEVICE_SINGLE_FP_CONFIG:            " << l_deviceSingleFpConfig << std::endl;
        
        /*
         ---------------------------------
         Device Quadro NVS 140M
         ---------------------------------
         CL_DEVICE_NAME:                       Quadro NVS 140M
         CL_DEVICE_VENDOR:                     NVIDIA Corporation
         CL_DRIVER_VERSION:                    260.99
         CL_DEVICE_VERSION:                    OpenCL 1.0 CUDA
         CL_DEVICE_TYPE:                       CL_DEVICE_TYPE_GPU
         CL_DEVICE_MAX_COMPUTE_UNITS:          2
         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:   3
         CL_DEVICE_MAX_WORK_ITEM_SIZES:        512 / 512 / 64
         CL_DEVICE_MAX_WORK_GROUP_SIZE:        512
         CL_DEVICE_MAX_CLOCK_FREQUENCY:        800 MHz
         CL_DEVICE_ADDRESS_BITS:               32
         CL_DEVICE_MAX_MEM_ALLOC_SIZE:         128 MByte
         CL_DEVICE_GLOBAL_MEM_SIZE:            113 MByte
         CL_DEVICE_ERROR_CORRECTION_SUPPORT:   no
         CL_DEVICE_LOCAL_MEM_TYPE:             local
         CL_DEVICE_LOCAL_MEM_SIZE:             16 KByte
         CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:   64 KByte
         CL_DEVICE_QUEUE_PROPERTIES:           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
         CL_DEVICE_QUEUE_PROPERTIES:           CL_QUEUE_PROFILING_ENABLE
         CL_DEVICE_IMAGE_SUPPORT:              1
         CL_DEVICE_MAX_READ_IMAGE_ARGS:        128
         CL_DEVICE_MAX_WRITE_IMAGE_ARGS:       8
         CL_DEVICE_SINGLE_FP_CONFIG:           INF-quietNaNs round-to-nearest round-to-zero round-to-inf fma
         */
    }
    
    std::cout << std::endl;
    
    /////////////////////////////////////
    //             CONTEXTE            //
    /////////////////////////////////////
    // Création du contexte
    *p_context = clCreateContext(NULL, l_numDevices, p_devices, NULL, NULL, &l_status);
    checkErr(l_status, "Impossible de créer un contexte à l'aide de clCreateContext");
    
    /////////////////////////////////////
    //         COMMAND QUEUE           //
    /////////////////////////////////////
    
    // Créer une command queue et l'associer au premier device (Celui qu'on veut utiliser)
    *p_cmdQueue = clCreateCommandQueue(*p_context, p_devices[0], 0, &l_status);
    checkErr(l_status, "Impossible de créer la command queue à l'aide de clCreateCommandQueue");
}

void compileProgram(cl_device_id *p_devices, cl_context p_context, cl_program *p_programme)
{
    cl_int l_status;
    
    char *source;
    const char *sourceFile = "gauss.cl";
    // Lecture du programme source dans le fichier gauss.cl
    source = readSource(sourceFile);
    
    // Créer le programme à partir du fichier source
    *p_programme = clCreateProgramWithSource(p_context, 1, (const char**)&source, NULL, &l_status);
    checkErr(l_status, "Impossible de créer le programme à partir de la source à l'aide de clCreateProgramWithSource");
    
    cl_int cl_buildErr;
    // Build (compile & link) the program for the devices.
    // Save the return value in 'buildErr' (the following
    // code will print any compilation errors to the screen)
    cl_buildErr = clBuildProgram(*p_programme, 0, p_devices, NULL, NULL, NULL);
    
    // If there are build errors, print them to the screen
    if (cl_buildErr != CL_SUCCESS) {
        std::cerr << "Impossible de construire le programme OpenCL." << cl_buildErr << std::endl;
        cl_build_status cl_buildStatus;
        for(unsigned int i = 0; i < 1; i++) {
            clGetProgramBuildInfo(*p_programme, p_devices[i], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &cl_buildStatus, NULL);
            if (cl_buildStatus == CL_SUCCESS) {
                continue;
            }
            
            char *buildLog;
            size_t buildLogSize;
            
            clGetProgramBuildInfo(*p_programme, p_devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
            
            buildLog = (char*)malloc(buildLogSize);
            if (buildLog == NULL) {
                std::cerr << "Impossible d'allouer l'espace pour les journaux de la construction du programme." << std::endl;
                exit(EXIT_FAILURE);
            }
            
            clGetProgramBuildInfo(*p_programme, p_devices[i], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
            buildLog[buildLogSize - 1] = '\0';
            
            std::cout << "Device " << i << " Build Log : " << std::endl << buildLog << std::endl;
            free(buildLog);
        }
        exit(EXIT_SUCCESS);
        
    }
    std::cout << "Programme OpenCL construit avec succès." << std::endl;
    
    free(source);
}

float trouverMax(float *tableauDeFloat, unsigned int lS, cl_context p_context, cl_program p_program, cl_command_queue p_cmdQueue) {
    
    cl_int l_status;
    static cl_mem cl_matriceReduce = NULL;
    static cl_mem cl_matriceSortieReduce = NULL;
    static cl_kernel cl_kernelReduce = NULL;
    
    float tableauFloatSortie[lS];
    
    if (tableauDeFloat != NULL) {
    
        if (cl_matriceReduce == NULL) {
            //cl_matriceReduce = clCreateBuffer(p_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, lS * sizeof(float), tableauDeFloat, &l_status);
            cl_matriceReduce = clCreateBuffer(p_context, CL_MEM_READ_ONLY, lS * sizeof(float), NULL, &l_status);
            checkErr(l_status, "Impossible de créer le buffer d'entrée de test pour la matrice à l'aide de clCreateBuffer");
        
            cl_matriceSortieReduce = clCreateBuffer(p_context, CL_MEM_READ_WRITE, lS * sizeof(float), NULL, &l_status);
            checkErr(l_status, "Impossible de créer le buffer de sortie de test pour la matrice à l'aide de clCreateBuffer");
        
        }
    
        /////////////////////////////////////
        //          KERNEL REDUCE          //
        /////////////////////////////////////
    
        if (cl_kernelReduce == NULL) {
            // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
            cl_kernelReduce = clCreateKernel(p_program, "reduce", &l_status);
            checkErr(l_status, "Impossible de créer le kernel cl_kernelReduce à partir du programme à l'aide de clCreateKernel");
    
            // Associer les paramètres aux buffres précédamment crées
            l_status  = clSetKernelArg(cl_kernelReduce, 0, sizeof(cl_mem), &cl_matriceReduce);
            checkErr(l_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
            l_status |= clSetKernelArg(cl_kernelReduce, 1, 32 * 1024, NULL);
            checkErr(l_status, "Impossible de créer l'argument 1 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
            l_status |= clSetKernelArg(cl_kernelReduce, 2, sizeof(unsigned int), &lS);
            checkErr(l_status, "Impossible de créer l'argument 2 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
            l_status |= clSetKernelArg(cl_kernelReduce, 3, sizeof(cl_mem), &cl_matriceSortieReduce);
            checkErr(l_status, "Impossible de créer l'argument 3 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        }
    
        size_t cl_ReduceProblemSize[1];
        cl_ReduceProblemSize[0] = lS;
    
        // Exécution du kernel
        //std::cout << std::endl << "Exécution du programme OpenCl sur le device avec les données : " << std::endl << "\t";
        //for (int i = 0 ; i < 25 ; i++)
        //{
        //    std::cout << tableauDeFloat[i] << " ";
        //}
    
        clEnqueueWriteBuffer(p_cmdQueue, cl_matriceReduce, CL_TRUE, 0, lS*sizeof(float), tableauDeFloat, 0, NULL, NULL);

        // Enfilement de la commande d'exécution du kernel
        l_status = clEnqueueNDRangeKernel(p_cmdQueue, cl_kernelReduce, 1, NULL, cl_ReduceProblemSize, NULL, 0, NULL, NULL);
        checkErr(l_status, "Impossible d'enfiler l'exécution du cl_kernelReduce sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
    
        // Read the OpenCL output buffer (d_C) to the host output array (C)
        clEnqueueReadBuffer(p_cmdQueue, cl_matriceSortieReduce, CL_TRUE, 0, lS * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
    
        //std::cout << std::endl << std::endl << "Résultat : " << std::endl << "\t";
        while (tableauFloatSortie[1] != 0)
        {
            //for (int i = 0 ; i < 25 ; i++)
            //{
            //    std::cout << tableauFloatSortie[i] << " ";
            //}
        
            //std::cout << " (Pas tout à fait réduit)" << std::endl;
        
            clEnqueueWriteBuffer(p_cmdQueue, cl_matriceReduce, CL_TRUE, 0, lS*sizeof(float), tableauFloatSortie, 0, NULL, NULL);
        
            l_status = clEnqueueNDRangeKernel(p_cmdQueue, cl_kernelReduce, 1, NULL, cl_ReduceProblemSize, NULL, 0, NULL, NULL);
            checkErr(l_status, "Impossible d'enfiler l'exécution du cl_kernelReduce sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
        
            clEnqueueReadBuffer(p_cmdQueue, cl_matriceSortieReduce, CL_TRUE, 0, lS*sizeof(float), tableauFloatSortie, 0, NULL, NULL);
        
        }
        
    } else {
        delete cl_matriceReduce;
        delete cl_matriceSortieReduce;
        delete cl_kernelReduce;
        
        return 0.0;
        
    }
    
    //std::cout << std::endl << "Highest value : " << tableauFloatSortie[0] << " ";
    
    //std::cout << std::endl << std::endl;
    
    return tableauFloatSortie[0];
}