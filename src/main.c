/**
 *
 * Simulation of 'geometric' security attack on the kkk protocol based on code provided
 * by Maduka Attamah
 *
 * Author: Maduka Attamah, Cameron Hargreaves
 *
 * Copyright 2016 Maduka Attamah
 */

#include "../lib/3kprotocolParallel.h"
#include <mpi.h>
#include <string.h>
#include <time.h>

#define EPOCH_LIMIT 15000000

int main(int argc, char *argv[]) {
    int comm_sz, myRank;
    int k = 0;
    int n = 0;
    int l = 0;

    int synchThreshold = 50;
    int my_synchTimeAB, my_synchTimeAC;
    int sumOfDifferenceAB, sumOfDifferenceAC, sumOfDifferenceBC;

    int aSeed, bSeed;

    char outputFlag[3];

    double timeStart, timeEnd;


    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Ensure the correct input arguments have been given and assign to variables, else exit
    if (argc == 6) {
        k = atoi(argv[1]);
        n = atoi(argv[2]);
        l = atoi(argv[3]);
        synchThreshold = atoi(argv[4]);
        strcpy(outputFlag, argv[5]);
    } else if (myRank == 0){
        printCLAError();
        return(1);
    }

    // Generate a seed in the first process to generate the random numbers for each of the inputs, then broadcast to each process
    if (myRank == 0) {
        aSeed = ((unsigned)time(NULL));
        bSeed = aSeed + 5;
    }

    MPI_Bcast(&aSeed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bSeed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // For each process generate a neural net using the same seed and the same structure
    struct NeuralNetwork neuralNetA = constructNeuralNetworkFromSeed(k, n, l, aSeed);
    struct NeuralNetwork neuralNetB = constructNeuralNetworkFromSeed(k, n, l, bSeed);

    // Generate a unique seed for each process for the C network and for each process create a network
    int myCSeed = (unsigned)time(NULL) + myRank * comm_sz;
    struct NeuralNetwork my_neuralNetC = constructNeuralNetworkFromSeed(k, n, l, myCSeed);

    /* Uncomment below block to get each process to print their initial A, B and C network weights */
    // printInitialWeights(comm_sz, myRank, k, n, neuralNetA, neuralNetB, my_neuralNetC);


    int** inputs = getRandomInputs(k, n);

    timeStart = MPI_Wtime();

     /*
     * To run the KKK Protocol normally (without any attacks), uncomment the call to function runKKKProtocol(...) below,
     * and comment out the call to function runGeometricAttackKKKProtocol(...) which follows.  To run the attack in 'offline'
     * mode the call to runGeometricAttackKKKProtocol(...) can come after the runKKKProtocol(...), but then you ought to modify
     * (or create another version of) the function runGeometricAttackKKKProtocol(...) to not update the weights of
     * neuralNetA and neuralNetB during the attack.
     */

    int epoch = 0;

    bool status = runGeometricAttackKKKProtocolParallelMPI(neuralNetA, neuralNetB, my_neuralNetC,
                                                            inputs, k, n, l, synchThreshold,
                                                            EPOCH_LIMIT, &epoch, myRank,
                                                            &my_synchTimeAB, &my_synchTimeAC);

    timeEnd = MPI_Wtime();

    sumOfDifferenceAB = compareWeights(neuralNetA, neuralNetB, k, n);
    sumOfDifferenceAC = compareWeights(neuralNetA, my_neuralNetC, k, n);
    sumOfDifferenceBC = compareWeights(neuralNetB, my_neuralNetC, k, n);


    for(int i = 0; i < comm_sz; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == myRank) {
            printFinal(outputFlag, status,
                       comm_sz, myRank,
                       epoch, EPOCH_LIMIT, synchThreshold,
                       my_synchTimeAB, my_synchTimeAC,
                       k, n, l,
                       sumOfDifferenceAB, sumOfDifferenceAC, sumOfDifferenceBC,
                       neuralNetA, neuralNetB, my_neuralNetC,
                       timeEnd - timeStart);
        }
    }

    freeMemoryForNetwork(neuralNetA, k, n);
    freeMemoryForNetwork(neuralNetB, k, n);
    freeMemoryForNetwork(my_neuralNetC, k, n);

    free(inputs);
    return 0;
}
