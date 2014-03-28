// System includes
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <sstream>
#include "Chrono.hpp"

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Signatures
char* readSource(const char *sourceFilename);
inline void checkErr(cl_int err, const char * texte);
float trouverMax(float *tableauDeFloat, unsigned int lineSize);
int trouverMaxPos(float *tableauDeFloat, unsigned int lineSize);
void diviserLigne(float *tableauDeFloat, float dividente, unsigned int lineSize);
void eliminerColonne(float *matriceRandom, float *matriceIdent, int matriceSize, int lineSize, int currentLine, int maxPos);
void initOpenCL();
void compileProgram();
std::string afficherTableau(float *tableau, unsigned int size);

// Variables OpenCL (Global afin de faciliciter les initialisations.
cl_platform_id *gPlatforms;
cl_device_id *gDevices;
cl_context gContext;
cl_command_queue gCmdQueue;
cl_program gProgramme;

// Kernel reduce max
cl_mem gBuffMatriceReduce = NULL;
cl_mem gBuffMatriceSortieReduce = NULL;
cl_kernel gKernelReduce = NULL;

// Kernel divide line
cl_mem gBuffMatriceDivide = NULL;
cl_kernel gKernelDivide = NULL;

// Kernel eliminer colonne
cl_mem gBuffMatriceEliminate;
cl_mem gBuffMatriceEliminateIdentity;
cl_mem gBuffMatriceEliminateLine;
cl_mem gBuffMatriceEliminateLineIdentity;
cl_kernel gKernelEliminate;
size_t cl_LocalWorkSize[3];

/*!
 * \brief Fonction principale, le travail est géré ici.
 */
int main(int argc, char ** argv)
{
    printf("Running Matrix Inversion program\n\n");
    Chrono monChrono;
    // Seed du random
    srand((unsigned int)time(NULL));
    
    // Taille de la matrice par défaut
    unsigned int lineSize = 2000;
    
    // local work size par défaut
    cl_LocalWorkSize[0] = 1024;
    
    if (argc == 3) {
        lineSize = atoi(argv[1]);
        cl_LocalWorkSize[0] = atoi(argv[2]);
        cl_LocalWorkSize[1] = cl_LocalWorkSize[0];
    }                                                               
    // Taille du tableau de la matrice
    unsigned int matriceSize = lineSize * lineSize;
    std::cout << "Taille du tableau : " << matriceSize << std::endl;
    
    // Initialisations d'OpenCL
    initOpenCL();
    
    /////////////////////////////////////
    //            PROGRAM              //
    /////////////////////////////////////
    // Compilation du programme
    compileProgram();

    // Deux tableaux, un pour la matrice généré aléatoirement, et un autre pour contenir la matrice identité qui subira toute les mêmes manipulations
    // que la matrice aléatoire.
    float *matriceRandom = new float[matriceSize];
    float *matriceInverse = (float*)calloc(matriceSize, sizeof(float));
    
    rand(); // Ne pas utiliser la première valeur du random, elle ne semble pas très aléatoire
    
    // Génération de la matrice aléatoire
    for (unsigned long long i = 0 ; i < matriceSize ; ++i) {
        matriceRandom[i] = rand() / (float)(RAND_MAX - 1);
        //matriceRandom[i] = (int)(rand() % 100) + 1;
        
    }
    
    // Construction de la matrice identité.
    for (unsigned long long i = 0 ; i < lineSize ; i++) {
        matriceInverse[i + lineSize * i] = 1;
    }
    
    float max;
    int maxPos;
    for (int k = 0 ; k < lineSize ; k++) {
        max = trouverMax(matriceRandom + lineSize * k, lineSize);
        
        maxPos = trouverMaxPos(matriceRandom + lineSize * k, lineSize);
        
        diviserLigne(matriceInverse + lineSize * k, matriceRandom[maxPos + lineSize * k], lineSize);
        float test[lineSize];
        
         // Swap colonnes, il serait peut-etre intéressant de paralléléser cette section.
        for (int i = 0 ; i < lineSize ; i++) {
            test[i] = matriceRandom[k + i * lineSize];
            matriceRandom[k + i * lineSize] = matriceRandom[maxPos + i * lineSize];
            matriceRandom[maxPos + i * lineSize] = test[i];
            
            test[i] = matriceInverse[k + i * lineSize];
            matriceInverse[k + i * lineSize] = matriceInverse[maxPos + i * lineSize];
            matriceRandom[maxPos + i * lineSize] = test[i];
        }
        
        diviserLigne(matriceRandom + lineSize * k, matriceRandom[maxPos + lineSize * k], lineSize);
        eliminerColonne(matriceRandom, matriceInverse, matriceSize, lineSize, k, maxPos);
        
    }
    
    // Afficher de l'information sur le work group size
    size_t work_group_size = 0;
    cl_int cl_status = clGetKernelWorkGroupInfo(gKernelDivide, gDevices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
    std::cout << "CL_KERNEL_WORK_GROUP_SIZE : " << work_group_size << std::endl;
    
    // Vérifier la marge d'erreur de la matrice inversé
    double sum = 0;
    for (int i = 0 ; i < matriceSize ; i++) {
        sum += matriceInverse[i];
    }
    
    // Affichage sur la sortie standard du résultat de l'exécution
    std::cout << "Matrice : " << lineSize << " x " << lineSize << std::endl;
    std::cout << "Erreur : " << sum - lineSize << std::endl;
    
    // Nettoyage
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
    
    // Temps total de l'exécution (Ce temps explose si on affiche les matrices en cours d'exéction)
    std::cout << "Duree : " << monChrono.get() << " secondes." << std::endl;
    
}

/*!
 * \brief Convertir une matrice en std::string.
 *
 * Cette fonction est principalement utilisé pour les tests afin de suivre ce que les différents kernels ont appliqués sur la matrice.
 * \param tableau Le tableau qu'on veut afficher
 * \param size Taille du tableau à convertir
 *
 * \return Une représentation std::string du tableau en entrée
 */
std::string afficherTableau(float *tableau, unsigned int size) {
    unsigned int lineSize = (unsigned int)sqrt(size);
    std::stringstream oss;
    
    // Forcer la précision du stringstream
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(6);
    
    // Parcourir le tableau afin de concatener toute les valeurs
    for (int i = 0 ; i < size ; i++) {
        if (i % lineSize == 0) {
            oss << "";
        }
        oss << tableau[i];
        if (i % lineSize != lineSize - 1) {
            oss << " ";
        } else {
            oss << "" << std::endl;
        }
    }
    return oss.str();
}

/*!
 * \brief Algo pour diviser tout les éléments d'une ligne par un nombre (le pivot)
 *
 * Première étape pout obtenir une matrice identité, faire disparaitre le plus grand nombre.
 *
 * \param tableauDeFloat Le tableau à réduire
 * \param dividente Tout les éléments du tableau seront divisés par cette valeur
 * \param lineSize La taille du tableau
 */
void diviserLigne(float *tableauDeFloat, float dividente, unsigned int lineSize) {
    cl_int cl_status;
    /////////////////////////////////////
    //            BUFFERS              //
    /////////////////////////////////////
    // Initialisation des buffers
    if (gBuffMatriceDivide == NULL) {
        gBuffMatriceDivide = clCreateBuffer(gContext, CL_MEM_READ_WRITE, lineSize * sizeof(float), NULL, &cl_status);
        checkErr(cl_status, "Impossible de créer le buffer d'entrée de test pour la matrice à l'aide de clCreateBuffer");
        
    }
    // Envoie des données à traiter dans le buffer du device
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceDivide, CL_TRUE, 0, lineSize * sizeof(float), tableauDeFloat, 0, NULL, NULL);
    /////////////////////////////////////
    //          KERNEL REDUCE          //
    /////////////////////////////////////
    // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
    if (gKernelDivide == NULL) {
        gKernelDivide = clCreateKernel(gProgramme, "divideVector", &cl_status);
        checkErr(cl_status, "Impossible de créer le kernel cl_kernelReduce à partir du programme à l'aide de clCreateKernel");
        
        // Associer le paramètre au kernel.
        cl_status  = clSetKernelArg(gKernelDivide, 0, sizeof(cl_mem), &gBuffMatriceDivide);
        checkErr(cl_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    }

    cl_status = clSetKernelArg(gKernelDivide, 1, sizeof(float), &dividente);
    checkErr(cl_status, "Impossible de créer l'argument 1a du cl_kernelReducea à partir du programme à l'aide de clSetKernelArg");
    
    size_t divideProblemSize[1];
    divideProblemSize[0] = cl_LocalWorkSize[0];//lineSize;
    // size_t cl_LocalWorkSize[1];
    // cl_LocalWorkSize[0] = 32;
    
    // Enfilement de la commande d'exécution du kernel
    cl_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelDivide, 1, NULL, divideProblemSize, cl_LocalWorkSize, 0, NULL, NULL);
    checkErr(cl_status, "Impossible d'enfiler l'exécution du cl_asdf sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
    
    // Lire le buffer de sortie dans la ligne du tableau en entrée.
    clEnqueueReadBuffer(gCmdQueue, gBuffMatriceDivide, CL_TRUE, 0, lineSize * sizeof(float), tableauDeFloat, 0, NULL, NULL);
}

/*!
 * \brief Exécution de la réduction proprement dite, on applique ici l'algorithme de gauss-jordan sur toute les lignes des matrices.
 *
 * Kernel qui fait le gros du travail de la réduction. L'algorithme est appliquer sur notre tableau généré ainsi que la matrice identité en parallel. 
 * De cette façon nous obtenons la matrice inversé dans un tableau séparé du tableau généré.
 *
 * \param matriceRandom La matrice généré aléatoirement
 * \param matriceIdent La matrice identité qui finira par devenir la matrice inversé
 * \param matriceSize La taille de la matrice (lineSize * lineSize)
 * \param lineSize La taille d'une ligne de la matrice
 * \param currentLine La ligne actuelle (étape k)
 * \param maxPos La position de l'élément le plus grand (pivot) Devrait normalement être == à k
 */
 
void eliminerColonne(float *matriceRandom, float *matriceIdent, int matriceSize, int lineSize, int currentLine, int maxPos) {
    
    cl_int cl_status;
    /////////////////////////////////////
    //            BUFFERS              //
    /////////////////////////////////////
    // Tableaux pour contenir les données à réduire.
    if (gBuffMatriceEliminate == NULL) {
        gBuffMatriceEliminate = clCreateBuffer(gContext, CL_MEM_READ_WRITE, matriceSize * sizeof(float), NULL, &cl_status);
        checkErr(cl_status, "Impossible de créer le buffer d'entrée gBuffMatriceEliminate pour la matrice à l'aide de clCreateBuffer");
        
        gBuffMatriceEliminateIdentity = clCreateBuffer(gContext, CL_MEM_READ_WRITE, matriceSize * sizeof(float), NULL, &cl_status);
        checkErr(cl_status, "Impossible de créer le buffer d'entrée gBuffMatriceEliminate pour la matrice à l'aide de clCreateBuffer");
        
        gBuffMatriceEliminateLine = clCreateBuffer(gContext, CL_MEM_READ_ONLY, lineSize * sizeof(float), NULL, &cl_status);
        checkErr(cl_status, "Impossible de créer le buffer d'entrée gBuffMatriceEliminateLine pour la matrice à l'aide de clCreateBuffer");
        
        gBuffMatriceEliminateLineIdentity = clCreateBuffer(gContext, CL_MEM_READ_ONLY, lineSize * sizeof(float), NULL, &cl_status);
    }
    
    // Écriture des données dans les buffers d'entrée (Transfert vers la mémoire du device.
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceEliminate, CL_TRUE, 0, matriceSize * sizeof(float), matriceRandom, 0, NULL, NULL);
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceEliminateIdentity, CL_TRUE, 0, matriceSize * sizeof(float), matriceIdent, 0, NULL, NULL);
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceEliminateLine, CL_TRUE, 0, lineSize * sizeof(float), matriceRandom + lineSize * currentLine, 0, NULL, NULL);
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceEliminateLineIdentity, CL_TRUE, 0, lineSize * sizeof(float), matriceIdent + lineSize * currentLine, 0, NULL, NULL);
    
    /////////////////////////////////////
    //          KERNEL REDUCE          //
    /////////////////////////////////////
    float *tabloTemp = new float[lineSize];
    float *tabloTempIdent = new float[lineSize];
    
    // Le kernel applique la réduction sur toute les lignes, il faut donc conserver la ligne actuelle dans des tableaux temporaires afin de pouvoir
    // les réaffecter dans le tableau
    for (int i = 0 ; i < lineSize ; i++) {
        tabloTemp[i] = matriceRandom[i + lineSize * currentLine];
        tabloTempIdent[i] = matriceIdent[i + lineSize * currentLine];
    } 
    
    // Si le kernel n'est pas initialisé, il faut l'initialiser
    if ( gKernelEliminate == NULL) {
        // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
        gKernelEliminate = clCreateKernel(gProgramme, "eliminateColumnVector", &cl_status);
        checkErr(cl_status, "Impossible de créer le kernel gKernelEliminate à partir du programme à l'aide de clCreateKernel");
        
        // Associate the input and output buffers with the kernel
        cl_status  = clSetKernelArg(gKernelEliminate, 0, sizeof(cl_mem), &gBuffMatriceEliminate);
        checkErr(cl_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        cl_status  = clSetKernelArg(gKernelEliminate, 1, sizeof(cl_mem), &gBuffMatriceEliminateIdentity);
        checkErr(cl_status, "Impossible de créer l'argument 0 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        cl_status |= clSetKernelArg(gKernelEliminate, 2, sizeof(int), &lineSize);
        checkErr(cl_status, "Impossible de créer l'argument 1b du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        cl_status |= clSetKernelArg(gKernelEliminate, 5, sizeof(cl_mem), &gBuffMatriceEliminateLine);
        checkErr(cl_status, "Impossible de créer l'argument 1 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        cl_status |= clSetKernelArg(gKernelEliminate, 6, sizeof(cl_mem), &gBuffMatriceEliminateLineIdentity);
        checkErr(cl_status, "Impossible de créer l'argument 1 du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
        
    }
            
    // Section des arguments variables du kernel
    cl_status = clSetKernelArg(gKernelEliminate, 3, sizeof(int), &maxPos);
    checkErr(cl_status, "Impossible de créer l'argument 1b du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    cl_status |= clSetKernelArg(gKernelEliminate, 4, sizeof(int), &currentLine);
    checkErr(cl_status, "Impossible de créer l'argument 1b du cl_kernelReduce à partir du programme à l'aide de clSetKernelArg");
    
    // Taille du problème
    size_t eliminateProblemSize[1];
    eliminateProblemSize[0] = cl_LocalWorkSize[0];//matriceSize;
    // size_t cl_LocalWorkSize[1];
    // cl_LocalWorkSize[0] = 32;
    
    // Enfilement de la commande d'exécution du kernel
    cl_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelEliminate, 1, NULL, eliminateProblemSize, cl_LocalWorkSize, 0, NULL, NULL);
    checkErr(cl_status, "Impossible d'enfiler l'exécution du cl_asdf sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
    
    // Lecture des buffers de sortie
    clEnqueueReadBuffer(gCmdQueue, gBuffMatriceEliminate, CL_TRUE, 0, matriceSize * sizeof(float), matriceRandom, 0, NULL, NULL);
    clEnqueueReadBuffer(gCmdQueue, gBuffMatriceEliminateIdentity, CL_TRUE, 0, matriceSize * sizeof(float), matriceIdent, 0, NULL, NULL);
    
    // On replace la ligne dans le tableau
    for (int i = 0 ; i < lineSize ; i++) {
        matriceRandom[i + lineSize * currentLine] = tabloTemp[i];
        matriceIdent[i + lineSize * currentLine] = tabloTempIdent[i];
    }
    
}

/*!
 * \brief Trouve la position de l'élément le plus grand du tableau fourni en paramètre.
 *
 * La fonction trouve le nombre le plus éloigné de 0 dans le tableau passé en paramètre.
 *
 * \param tableauDeFloat Le tableau en question
 * \param lineSize Taille du tableau
 *                                             
 * \return L'emplacement de l'élément le plus grand (ou plus petit selon la valeur absolue)
 * \todo Créer un kernel pour cette section
 */
int trouverMaxPos(float *tableauDeFloat, unsigned int lineSize) {
    
    float currentMax = -INFINITY;
    int maxPos = -1;
    for (int i = 0 ; i < lineSize ; i++) {
        float tmp = fabsf(tableauDeFloat[i]);
        if (fabsf(tableauDeFloat[i]) > currentMax) {
            currentMax = fabsf(tableauDeFloat[i]);
            maxPos = i;
        }
    }
    return maxPos;
}
     
/*!
 * \brief Trouve le nombre maximum d'un tableau passé en paramètre
 *
 * Cette fonction prend un tableau d'une longueur arbitraire et exécute un kernel de réduction max plusieurs fois afin de trouver le nombre le plus grand.
 *
 * \param tableauDeFloat Le tableau sur lequel on veut appliquer le kernel
 * \param lineSize La taille du tableau passé en paramètre
 *
 * \return L'élément le plus grand du tableau
 *
 * \author Giovanni Victorette
 * \author Jimmy St-Hilaire
 */
float trouverMax(float *tableauDeFloat, unsigned int lineSize) {
    
    cl_int l_status;
    
    long nextPowerOfTwo = 1;
    // Trouver la prochaine puissance de deux selon la taille du tableau
    while (nextPowerOfTwo < lineSize) {
        nextPowerOfTwo = nextPowerOfTwo << 1;
    }
    
    float paddedTableauDeFloat[nextPowerOfTwo];
    float tableauFloatSortie[nextPowerOfTwo];
    
    // On pad le tableau puisque le kernel nécessite un tableau d'une taille de puissance 2
    for (int i = 0 ; i < nextPowerOfTwo ; i++) {
        if (i < lineSize) {
            paddedTableauDeFloat[i] = tableauDeFloat[i];
            
        } else {
            paddedTableauDeFloat[i] = -INFINITY;
        }
    }
    
    // Créer les buffers si ils n'ont pas été initialisé auparavent
    if (gBuffMatriceReduce == NULL) {
        gBuffMatriceReduce = clCreateBuffer(gContext, CL_MEM_READ_ONLY, nextPowerOfTwo * sizeof(float), NULL, &l_status);
        checkErr(l_status, "Impossible de créer le buffer d'entrée de test pour la matrice à l'aide de clCreateBuffer");
            
        gBuffMatriceSortieReduce = clCreateBuffer(gContext, CL_MEM_READ_WRITE, nextPowerOfTwo * sizeof(float), NULL, &l_status);
        checkErr(l_status, "Impossible de créer le buffer de sortie de test pour la matrice à l'aide de clCreateBuffer");
            
    }
        
    /////////////////////////////////////
    //          KERNEL REDUCE          //
    /////////////////////////////////////
    if (gKernelReduce == NULL) {
        // Création du kernel pour trouver le nombre maximal d'une ligne (pivot)
        gKernelReduce = clCreateKernel(gProgramme, "reduce", &l_status);
        checkErr(l_status, "Impossible de créer le kernel cl_kernelReduce?? à partir du programme à l'aide de clCreateKernel");
            
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
    cl_ReduceProblemSize[0] = cl_LocalWorkSize[0];//nextPowerOfTwo;
    // size_t cl_LocalWorkSize[1];
    // cl_LocalWorkSize[0] = 32;
        
    // Exécution du kernel
    clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceReduce, CL_TRUE, 0, nextPowerOfTwo * sizeof(float), paddedTableauDeFloat, 0, NULL, NULL);
        
    // Enfilement de la commande d'exécution du kernel
    l_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelReduce, 1, NULL, cl_ReduceProblemSize, cl_LocalWorkSize, 0, NULL, NULL);
    checkErr(l_status, "Impossible d'enfiler l'exécution du cl_kernelReduce1 sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
        
    // Lecture du résultat de l'exécution du kernel
    clEnqueueReadBuffer(gCmdQueue, gBuffMatriceSortieReduce, CL_TRUE, 0, nextPowerOfTwo * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
        
    // Si le tableau est excessivement grand, il faut faire plusieurs appels au kernel afin de réduire le tableau jusqu'a ce qu'une seule valeur ne soit restante
    while (tableauFloatSortie[1] != 0)
    {
        clEnqueueWriteBuffer(gCmdQueue, gBuffMatriceReduce, CL_TRUE, 0, nextPowerOfTwo * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
            
        l_status = clEnqueueNDRangeKernel(gCmdQueue, gKernelReduce, 1, NULL, cl_ReduceProblemSize, NULL, 0, NULL, NULL);
        checkErr(l_status, "Impossible d'enfiler l'exécution du cl_kernelReduce2 sur cl_cmdQueue à l'aide de clEnqueueNDRangeKernel");
            
        clEnqueueReadBuffer(gCmdQueue, gBuffMatriceSortieReduce, CL_TRUE, 0, nextPowerOfTwo * sizeof(float), tableauFloatSortie, 0, NULL, NULL);
            
    }
    
    return tableauFloatSortie[0];
}

/*!
 * \brief Function utilisé pour lire le code source des fonctions OpenCL     
 *
 * Cette fonction retourne un pointeur sur une chaîne de caractêre C représentant le code source de l'Application OpenCL.
 *
 * \author Giovanni Victorette
 * \author Jimmy St-Hilaire
 *
 * \param sourceFilename Le fichier à lire
 * \return Une chaîne C qui représente le fichier lu.
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

/*!
 * \brief Initialise l'environnement OpenCL
 *
 * Cette fonction initialise l'environnement OpenCL et affiche certaines informations concernant celui-ci, 
 * elle se concentre sur la première plateforme ainsi que la premiere device.
 *
 * \author Giovanni Victorette
 * \author Jimmy St-Hilaire
 */
void initOpenCL() {
    cl_int l_status;
    cl_uint l_numPlatforms;
    
    cl_uint l_numDevices = 0;
    
    /////////////////////////////////////
    //          PLATEFORM              //
    /////////////////////////////////////
    // Obtention du nombre de plateforme
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
        cl_uint l_deviceAddresBits;
        char l_deviceSingleFpConfig[100];
        
        // Plus d'options existe
        clGetDeviceInfo(gDevices[i], CL_DEVICE_NAME, sizeof(l_deviceName), l_deviceName, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_VENDOR, sizeof(l_deviceVendor), l_deviceVendor, NULL);
        clGetDeviceInfo(gDevices[i], CL_DRIVER_VERSION, sizeof(l_driverVersion), l_driverVersion, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_VERSION, sizeof(l_deviceVersion), l_deviceVersion, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_TYPE, sizeof(l_deviceType), l_deviceType, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(l_deviceMaxComputeUnits), &l_deviceMaxComputeUnits, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(l_deviceMaxWorkGroupSize), &l_deviceMaxWorkGroupSize, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(l_deviceSingleFpConfig), &l_deviceSingleFpConfig, NULL);
        clGetDeviceInfo(gDevices[i], CL_DEVICE_ADDRESS_BITS, sizeof(l_deviceAddresBits), &l_deviceAddresBits, NULL);
    
        std::cout << "\tCL_DEVICE_NAME:                        " << l_deviceName << std::endl;
        std::cout << "\tCL_DEVICE_VENDOR:                      " << l_deviceVendor << std::endl;
        std::cout << "\tCL_DRIVER_VERSION:                     " << l_driverVersion << std::endl;
        std::cout << "\tCL_DEVICE_VERSION:                     " << l_deviceVersion << std::endl;
        std::cout << "\tCL_DEVICE_MAX_COMPUTE_UNITS:           " << l_deviceMaxComputeUnits << std::endl;
        std::cout << "\tCL_DEVICE_MAX_WORK_GROUP_SIZE:         " << l_deviceMaxWorkGroupSize << std::endl;
        std::cout << "\tCL_DEVICE_SINGLE_FP_CONFIG:            " << l_deviceSingleFpConfig << std::endl;
        std::cout << "\tCL_DEVICE_ADDRESS_BITS:                " << l_deviceAddresBits << std::endl;
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

/*!
 * \brief Fonction utilisé pour compiler le programme OpenCL
 *
 * La fonction recherche un fichier nommé "gauss.cl" et le compile dynamiquement.
 * \pre L'environnement OpenCL doit avoir été initialisé préalablement
 * \author Giovanni Victorette
 * \author Jimmy St-Hilaire
 */
void compileProgram()
{
    cl_int l_status = 0;
    
    // Code source du programme OpenCL
    char *source;
    const char *sourceFile = "gauss.cl";
    
    // Lecture du programme source dans le fichier gauss.cl
    source = readSource(sourceFile);
    const size_t *length = {0};
    
    // Créer le programme à partir du fichier source
    gProgramme = clCreateProgramWithSource(gContext, 1, (const char**)&source, length, &l_status);
    checkErr(l_status, "Impossible de créer le programme à partir de la source1 à l'aide de clCreateProgramWithSource");
    
    cl_int cl_buildErr;
    // Compile et lie le programme
    cl_buildErr = clBuildProgram(gProgramme, 1, gDevices, NULL, NULL, NULL);
    
    // Si il existe des erreurs de compilation du programme OpenCL
    if (cl_buildErr != CL_SUCCESS) {
        std::cerr << "Impossible de construire le programme OpenCL." << cl_buildErr << std::endl;
        cl_build_status cl_buildStatus;
        
        // Itération sur les devices pour afficher toute les erreurs (Ici seulement la première)
        for(unsigned int i = 0; i < 1; i++) {
            clGetProgramBuildInfo(gProgramme, gDevices[i], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &cl_buildStatus, NULL);
            if (cl_buildStatus == CL_SUCCESS) {
                continue;
            }
            
            char *buildLog;
            size_t buildLogSize;
            
            // Obtention de la taille des logs d'erreurs de lac ompilation
            clGetProgramBuildInfo(gProgramme, gDevices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
            
            // Allouer l'espace pour contenir les logs
            buildLog = (char*)malloc(buildLogSize);
            if (buildLog == NULL) {
                std::cerr << "Impossible d'allouer l'espace pour les journaux de la construction du programme." << std::endl;
                exit(EXIT_FAILURE);
            }
            
            // Obtention des logs
            clGetProgramBuildInfo(gProgramme, gDevices[i], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
            buildLog[buildLogSize - 1] = '\0';
            
            std::cout << "Device " << i << " Build Log : " << std::endl << buildLog << std::endl;
            free(buildLog);
        }
        // On termine le programme proprement
        exit(EXIT_SUCCESS);
        
    }
    std::cout << "Programme OpenCL construit avec succès." << std::endl;
    
    free(source);
}

