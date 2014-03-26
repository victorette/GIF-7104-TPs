// System includes
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <sstream>

// OpenCL includes
#include <OpenCL/opencl.h>

// Signatures
char* readSource(const char *sourceFilename);
inline void checkErr(cl_int err, const char * texte);
float trouverMax(float *tableauDeFloat, unsigned int lS);
void initOpenCL();
void compileProgram();
std::string afficherTableau(float *tableau, unsigned int size);

cl_platform_id *gPlatforms;
cl_device_id *gDevices;
cl_context gContext;
cl_command_queue gCmdQueue;
cl_program gProgramme;

// Kernel Reduce Max
cl_mem gBuffMatriceReduce = NULL;
cl_mem gBuffMatriceSortieReduce = NULL;
cl_kernel gKernelReduce = NULL;

// Kernel Diviser Ligne

/*!
 * \brief Moo!
 */
int main(int argc, char ** argv)
{
    printf("Running Matrix Inversion program\n\n");
    
    srand((unsigned int)time(NULL));
    
    unsigned int lineSize = 12;
    unsigned int linePlusIdentitySize = lineSize * 2;
    unsigned int matriceSize = lineSize * lineSize;
    std::cout << "Taille du tableau : " << matriceSize << std::endl;
    
    // Initialisation des variables OpenCL
    cl_int cl_status;  // use as return value for most OpenCL functions
    
    initOpenCL();
    
    /////////////////////////////////////
    //            PROGRAM              //
    /////////////////////////////////////
    compileProgram();
    
    float *matriceRandom = new float[matriceSize];
    
    for (unsigned long long i = 0; i < matriceSize ; ++i) {
        matriceRandom[i] = rand() / (float)RAND_MAX;
        //matriceRandom[i] = (int)(rand() % lineSize) + 1;
        
    }
    std::cout << afficherTableau(matriceRandom, matriceSize);
    std::cout << std::endl;
    
    float max = trouverMax(matriceRandom, lineSize);
    std::cout << std::endl << "Max : " << max << std::endl;
    
    /////////////////////////////////////
    //            BUFFERS              //
    /////////////////////////////////////
    // Tableaux pour contenir les données à réduire.
    
    cl_mem gBuffMatriceDivide;
    gBuffMatriceDivide = clCreateBuffer(gContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, lineSize * sizeof(float), matriceRandom, &cl_status);
    checkErr(cl_status, "Impossible de créer le buffer d'entrée de test pour la matrice à l'aide de clCreateBuffer");
    
    /////////////////////////////////////
    //          KERNEL REDUCE          //
    /////////////////////////////////////
    cl_kernel gKernelDivide;
    
    // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
    gKernelDivide = clCreateKernel(gProgramme, "divideVector", &cl_status);
    checkErr(cl_status, "Impossible de créer le kernel cl_kernelReduce à partir du programme à l'aide de clCreateKernel");
    
    // Associate the input and output buffers with the kernel
    cl_status  = clSetKernelArg(gKernelDivide, 0, sizeof(cl_mem), &gBuffMatriceDivide);
    checkErr(cl_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    cl_status |= clSetKernelArg(gKernelDivide, 1, sizeof(float), &max);
    checkErr(cl_status, "Impossible de créer l'argument 1 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    
    size_t cl_ReduceProblemSize[1];
    cl_ReduceProblemSize[0] = lineSize;
    
    // Enfilement de la commande d'exécution du kernel
    cl_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelDivide, 1, NULL, cl_ReduceProblemSize, NULL, 0, NULL, NULL);
    checkErr(cl_status, "Impossible d'enfiler l'exécution du cl_asdf sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
    
    // Read the OpenCL output buffer (d_C) to the host output array (C)
    clEnqueueReadBuffer(gCmdQueue, gBuffMatriceDivide, CL_TRUE, 0, lineSize * sizeof(float), matriceRandom, 0, NULL, NULL);
    
    std::cout << afficherTableau(matriceRandom, matriceSize);
    std::cout << std::endl;
    
    
    clReleaseKernel(gKernelReduce);
    clReleaseKernel(gKernelDivide);
    clReleaseProgram(gProgramme);
    clReleaseCommandQueue(gCmdQueue);
    clReleaseMemObject(gBuffMatriceReduce);
    clReleaseMemObject(gBuffMatriceSortieReduce);
    clReleaseMemObject(gBuffMatriceDivide);
    clReleaseContext(gContext);
    
    
    free(gDevices);
    free(gPlatforms);
    
    
}

std::string afficherTableau(float *tableau, unsigned int size) {
    unsigned int lineSize = (unsigned int)sqrt(size);
    std::stringstream oss;
    
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(4);
    
    for (int i = 0 ; i < size ; i++) {
        if (i % lineSize == 0) {
            oss << "[";
        }
        oss << tableau[i];
        if (i % lineSize != lineSize - 1) {
            oss << ", ";
        } else {
            oss << "]" << std::endl;
        }
    }
    return oss.str();
}

float trouverMax(float *tableauDeFloat, unsigned int lineSize) {
    
    cl_int l_status;
    
    // Kernel Reduce Max
    /*
     cl_mem gBuffMatriceReduce = NULL;
     cl_mem gBuffMatriceSortieReduce = NULL;
     cl_kernel gKernelReduce = NULL;
     */
    
    float tableauFloatSortie[lineSize];
    
    if (gBuffMatriceReduce == NULL) {
        gBuffMatriceReduce = clCreateBuffer(gContext, CL_MEM_READ_ONLY, lineSize * sizeof(float), NULL, &l_status);
        checkErr(l_status, "Impossible de créer le buffer d'entrée de test pour la matrice à l'aide de clCreateBuffer");
        
        gBuffMatriceSortieReduce = clCreateBuffer(gContext, CL_MEM_READ_WRITE, lineSize * sizeof(float), NULL, &l_status);
        checkErr(l_status, "Impossible de créer le buffer de sortie de test pour la matrice à l'aide de clCreateBuffer");
        
    }
    
    /////////////////////////////////////
    //          KERNEL REDUCE          //
    /////////////////////////////////////
    
    if (gKernelReduce == NULL) {
        // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
        gKernelReduce = clCreateKernel(gProgramme, "reduce", &l_status);
        checkErr(l_status, "Impossible de créer le kernel cl_kernelReduce à partir du programme à l'aide de clCreateKernel");
        
        // Associer les paramètres aux buffres précédamment crées
        l_status  = clSetKernelArg(gKernelReduce, 0, sizeof(cl_mem), &gBuffMatriceReduce);
        checkErr(l_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        l_status |= clSetKernelArg(gKernelReduce, 1, 32 * 1024, NULL);
        checkErr(l_status, "Impossible de créer l'argument 1 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        l_status |= clSetKernelArg(gKernelReduce, 2, sizeof(unsigned int), &lineSize);
        checkErr(l_status, "Impossible de créer l'argument 2 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        l_status |= clSetKernelArg(gKernelReduce, 3, sizeof(cl_mem), &gBuffMatriceSortieReduce);
        checkErr(l_status, "Impossible de créer l'argument 3 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    }
    
    size_t cl_ReduceProblemSize[1];
    cl_ReduceProblemSize[0] = lineSize;
    
    // Exécution du kernel
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceReduce, CL_TRUE, 0, lineSize * sizeof(float), tableauDeFloat, 0, NULL, NULL);
    
    // Enfilement de la commande d'exécution du kernel
    l_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelReduce, 1, NULL, cl_ReduceProblemSize, NULL, 0, NULL, NULL);
    checkErr(l_status, "Impossible d'enfiler l'exécution du cl_kernelReduce1 sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
    
    // Read the OpenCL output buffer (d_C) to the host output array (C)
    clEnqueueReadBuffer(gCmdQueue, gBuffMatriceSortieReduce, CL_TRUE, 0, lineSize * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
    
    while (tableauFloatSortie[1] != 0)
    {
        clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceReduce, CL_TRUE, 0, lineSize * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
        
        l_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelReduce, 1, NULL, cl_ReduceProblemSize, NULL, 0, NULL, NULL);
        checkErr(l_status, "Impossible d'enfiler l'exécution du cl_kernelReduce2 sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
        
        clEnqueueReadBuffer(gCmdQueue, gBuffMatriceSortieReduce, CL_TRUE, 0, lineSize * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
        
    }
    
    return tableauFloatSortie[0];
}

/*!
 * \brief Function utilisé pour lire le code source des fonctions OpenCL
 * \author Giovanni Victorette
 */
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

/*!
 * \brief Moo2!
 */
inline void checkErr(cl_int err, const char * texte)
{
    if (err != CL_SUCCESS) {
        std::cerr << "Erreur : " << texte
        << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void initOpenCL() {
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
    gPlatforms = (cl_platform_id*)malloc(l_numPlatforms * sizeof(cl_platform_id));
    if (gPlatforms == NULL) {
        std::cerr << "Impossible d'allouer l'espace pour toute les plateformes." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Obtention des plateformes
    l_status = clGetPlatformIDs(l_numPlatforms, gPlatforms, NULL);
    checkErr(l_status, "Impossible d'obtenir les plateformes à l'aide de clGetPlatformIDs");
    
    // Affichage des informations de base de la plateforme
    std::cout << l_numPlatforms << " plateformes trouvée." << std::endl;
    for (unsigned int i = 0; i < l_numPlatforms; i++) {
        char buf[100];
        std::cout << "Plateforme " << i << std::endl;
        
        l_status = clGetPlatformInfo(gPlatforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
        checkErr(l_status, "Impossible d'obtenir le vendeur de la plateforme à l'aide de clGetPlatformInfo");
        std::cout << "\tVendor : " << buf << std::endl;
        
        l_status = clGetPlatformInfo(gPlatforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        checkErr(l_status, "Impossible d'obtenir le nom de la plateforme à l'aide de clGetPlatformInfo");
        std::cout << "\tName : " << buf << std::endl;
        
    }
    
    std::cout << std::endl;
    
    /////////////////////////////////////
    //             DEVICE              //
    /////////////////////////////////////
    
    // Obtenir le nombre de devices
    l_status = clGetDeviceIDs(gPlatforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &l_numDevices);
    checkErr(l_status, "Impossible d'obtenir le nombre de devices à l'aide de clGetDeviceIDs");
    
    // Make sure some devices were found
    if (l_numDevices == 0) {
        std::cerr << "Aucune devices retrouvé." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Allouer de la mémoire pour toute les devices
    gDevices = (cl_device_id*)malloc(l_numDevices * sizeof(cl_device_id));
    if (gDevices == NULL) {
        std::cerr << "Impossible d'allouer l'espace pour toute les devices." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Obtenir les devices
    l_status = clGetDeviceIDs(gPlatforms[0], CL_DEVICE_TYPE_GPU, l_numDevices, gDevices, NULL);
    checkErr(l_status, "Impossible d'obtenir les devices à l'aide de clGetDeviceIDs");
    
    // Afficher des informations de base à propos des devices
    std::cout << l_numDevices << " devices trouvé" << std::endl;
    for (unsigned int i = 0; i < l_numDevices; i++) {
        char buf[100];
        std::cout << "Device " << i << std::endl;
        l_status = clGetDeviceInfo(gDevices[i], CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
        std::cout << "\tDevice : " << buf << std::endl;
        checkErr(l_status, "Impossible d'obtenir le vendeur de la device à l'aide de clGetDeviceInfo");
        
        l_status = clGetDeviceInfo(gDevices[i], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
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
        
        clGetDeviceInfo(gDevices[i], CL_DEVICE_NAME, sizeof(l_deviceName), l_deviceName, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_VENDOR, sizeof(l_deviceVendor), l_deviceVendor, NULL);
        clGetDeviceInfo(gDevices[i], CL_DRIVER_VERSION, sizeof(l_driverVersion), l_driverVersion, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_VERSION, sizeof(l_deviceVersion), l_deviceVersion, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_TYPE, sizeof(l_deviceType), l_deviceType, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(l_deviceMaxComputeUnits), &l_deviceMaxComputeUnits, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(l_deviceMaxWorkGroupSize), &l_deviceMaxWorkGroupSize, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(l_deviceSingleFpConfig), &l_deviceSingleFpConfig, NULL);
        
        std::cout << "\tCL_DEVICE_NAME:                        " << l_deviceName << std::endl;
        std::cout << "\tCL_DEVICE_VENDOR:                      " << l_deviceVendor << std::endl;
        std::cout << "\tCL_DRIVER_VERSION:                     " << l_driverVersion << std::endl;
        std::cout << "\tCL_DEVICE_VERSION:                     " << l_deviceVersion << std::endl;
        std::cout << "\tCL_DEVICE_MAX_COMPUTE_UNITS:           " << l_deviceMaxComputeUnits << std::endl;
        std::cout << "\tCL_DEVICE_MAX_WORK_GROUP_SIZE:         " << l_deviceMaxWorkGroupSize << std::endl;
        std::cout << "\tCL_DEVICE_SINGLE_FP_CONFIG:            " << l_deviceSingleFpConfig << std::endl;
    }
    
    std::cout << std::endl;
    
    /////////////////////////////////////
    //             CONTEXTE            //
    /////////////////////////////////////
    // Création du contexte
    gContext = clCreateContext(NULL, l_numDevices, gDevices, NULL, NULL, &l_status);
    checkErr(l_status, "Impossible de créer un contexte à l'aide de clCreateContext");
    
    /////////////////////////////////////
    //         COMMAND QUEUE           //
    /////////////////////////////////////
    
    // Créer une command queue et l'associer au premier device (Celui qu'on veut utiliser)
    gCmdQueue = clCreateCommandQueue(gContext, gDevices[0], 0, &l_status);
    checkErr(l_status, "Impossible de créer la command queue à l'aide de clCreateCommandQueue");
}

void compileProgram()
{
    cl_int l_status = 0;
    
    char *source;
    const char *sourceFile = "gauss.cl";
    // Lecture du programme source dans le fichier gauss.cl
    source = readSource(sourceFile);
    const size_t *length = {0};
    // Créer le programme à partir du fichier source
    gProgramme = clCreateProgramWithSource(gContext, 1, (const char**)&source, length, &l_status);
    checkErr(l_status, "Impossible de créer le programme à partir de la source1 à l'aide de clCreateProgramWithSource");
    
    cl_int cl_buildErr;
    // Build (compile & link) the program for the devices.
    // Save the return value in 'buildErr' (the following
    // code will print any compilation errors to the screen)
    cl_buildErr = clBuildProgram(gProgramme, 1, gDevices, NULL, NULL, NULL);
    
    // If there are build errors, print them to the screen
    if (cl_buildErr != CL_SUCCESS) {
        std::cerr << "Impossible de construire le programme OpenCL." << cl_buildErr << std::endl;
        cl_build_status cl_buildStatus;
        for(unsigned int i = 0; i < 1; i++) {
            clGetProgramBuildInfo(gProgramme, gDevices[i], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &cl_buildStatus, NULL);
            if (cl_buildStatus == CL_SUCCESS) {
                continue;
            }
            
            char *buildLog;
            size_t buildLogSize;
            
            clGetProgramBuildInfo(gProgramme, gDevices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
            
            buildLog = (char*)malloc(buildLogSize);
            if (buildLog == NULL) {
                std::cerr << "Impossible d'allouer l'espace pour les journaux de la construction du programme." << std::endl;
                exit(EXIT_FAILURE);
            }
            
            clGetProgramBuildInfo(gProgramme, gDevices[i], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
            buildLog[buildLogSize - 1] = '\0';
            
            std::cout << "Device " << i << " Build Log : " << std::endl << buildLog << std::endl;
            free(buildLog);
        }
        exit(EXIT_SUCCESS);
        
    }
    std::cout << "Programme OpenCL construit avec succès." << std::endl;
    
    free(source);
}

