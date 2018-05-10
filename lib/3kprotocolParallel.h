/**
*
*   C Library for the implementation and testing of the geometric attack
*   of the 3K protocol
*
*   Author: Cameron Hargreaves
*
* Adapted by code provided by Maduka Attamah
*
* Copyright 2016-2017 Maduka Attamah
*
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "3kprotocol.h"

bool runGeometricAttackKKKProtocolParallelMPI(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork attackerNet, int** inputs, int k, int n, int l, int syncThreshold, int epochLimit, int* epochFinal, int myRank, int *synchTimeAB, int *synchTimeAC);
void initWeightsFromArray(int** weights, int k, int n, int l, int** existingWeights);
struct NeuralNetwork constructNetworkFromWeights(int k, int n, int l, int** existingWeights);
int compareWeights(struct NeuralNetwork neuralNet1, struct NeuralNetwork neuralNet2, int k, int n);
void initWeightsFromSeed(int** weights, int k, int n, int l, int seed);
void printInitialWeights(int comm_sz, int myRank, int k, int n, struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork my_neuralNetC);
void printFinal(char outputFlag[3], bool status, int comm_sz, int myRank, int epoch, int EPOCH_LIMIT, int synchThreshold, int my_synchTimeAB, int my_synchTimeAC, int k, int n, int l, int sumOfDifferenceAB, int sumOfDifferenceAC, int sumOfDifferenceBC, struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork my_neuralNetC, double wallTime);

void printCLAError() {
    printf("Incorrect number of commandline arguments given, please enter\n");
    printf("[1] k number of perceptrons\n");
    printf("[2] n number of inputs to each perceptron\n");
    printf("[3] l range of weights for each perceptron\n");
    printf("[4] synch threshold for A and B to synchronise\n");
    printf("[5] flag for output format to be in, -v for verbose, -s for short, -p for pythonic, -c for csv\n");
}

bool runGeometricAttackKKKProtocolParallelMPI(struct NeuralNetwork neuralNetA,
                                              struct NeuralNetwork neuralNetB,
                                              struct NeuralNetwork attackerNet,
                                              int** inputs,
                                              int k,
                                              int n,
                                              int l,
                                              int syncThreshold,
                                              int epochLimit,
                                              int* epochFinal,
                                                  int myRank,
                                              int *synchTimeAB,
                                              int *synchTimeAC) {
    int my_synchronisationAB = 0;
    int my_synchronisationAC = 0;
    int epoch = 0;


    while ((my_synchronisationAB < syncThreshold) && (epoch < epochLimit)) {
        int outputA = getNetworkOutput(neuralNetA, inputs, k, n);
        int outputB = getNetworkOutput(neuralNetB, inputs, k, n);
        int outputC = getNetworkOutput(attackerNet, inputs, k, n);

        if ((outputA == outputB) && (outputA == outputC)) {
            //Increase synchronisation count, s.
            my_synchronisationAB = my_synchronisationAB + 1;
            my_synchronisationAC = my_synchronisationAC + 1;

            //Update the weights of all three networks using the anti-Hebbian learning rule.
            updateWeights(neuralNetA, inputs, k, n, l);
            updateWeights(neuralNetB, inputs, k, n, l);
            updateWeights(attackerNet, inputs, k, n, l);

        } else if ((outputA == outputB) && (outputA != outputC)) {
            //Reset the synchronisation count for A and C, however update the synchronisation for A and B
            my_synchronisationAC = 0;
            my_synchronisationAB = my_synchronisationAB + 1;

            // Get the hidden neuron in C (attackerNet) for which the sum of its weights * inputs is minimised.
            int kthHidden = getMinInputSumNeuron(attackerNet, inputs, k, n);

            //Negate the output of the "minimum sum neuron" obtained above.
            attackerNet.hiddenLayerOutputs[kthHidden] = attackerNet.hiddenLayerOutputs[kthHidden] * (-1);

            //Now update the weights of C with the new hidden "bits" (outputs).
            updateWeights(attackerNet, inputs, k, n, l);

            //Update A and B's weights too, using the  anti-Hebbian learning rule.
            updateWeights(neuralNetA, inputs, k, n, l);
            updateWeights(neuralNetB, inputs, k, n, l);

        } else if ((outputA != outputB) && (outputA == outputC)) {
            // Update the synch time of A and C only
            my_synchronisationAC = my_synchronisationAC + 1;

        } else {
            //Reset the synchronisation count - there was no synchronisation or sychronisation broke down in the round.
            my_synchronisationAB = 0;
            my_synchronisationAC = 0;
        }
        //Prepare for next round.
        free(inputs);
        //Get new random inputs for the next round.
        inputs = getRandomInputs(k, n);

        //Increment the round count. We will not run the protocol for ever - we will stop after a predefined number of rounds if
        //synchronisation has not been reached by then.
        epoch = epoch + 1;
    }

    *synchTimeAB = my_synchronisationAB;
    *synchTimeAC = my_synchronisationAC;
    *epochFinal = epoch;

    //Did the simulation end due to reach of synchronisation threshold?
    if (my_synchronisationAB == syncThreshold) {
        return true;
    }
    return false;
}

/**
 * Constructs a new two layered neural network with k perceptrons, n inputs per perceptron and weight across each input generated randomly
 * from the range -l to l.
 * @param k
 * @param n
 * @param l
 * @return the newly constructed neural network.
 */
struct NeuralNetwork constructNeuralNetworkFromSeed(int k, int n, int l, int seed) {
    struct NeuralNetwork neuralNetwork;
    // Allocate memory block for the neural network weights;
    neuralNetwork.weights = malloc(sizeof (int*) * (k));

    for (int i = 0; i < k; i++) {
        neuralNetwork.weights[i] = malloc(sizeof (int) * n);
    }

    //Allocate memory blocks for the hidden layer outputs.
    neuralNetwork.hiddenLayerOutputs = malloc(sizeof (int) * k);
    initWeightsFromSeed(neuralNetwork.weights, k, n, l, seed);

    return neuralNetwork;
}

/**
 * Get semi random weights (-l to l) for a neural network with k perceptrons and n inputs per
 * perceptron by setting the same seed so that we can share across processes
 * @param weights
 * @param k
 * @param n
 * @param l
 */
void initWeightsFromSeed(int** weights, int k, int n, int l, int seed) {
    srand(seed);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            weights[i][j] = weightRand(l);
        }
    }
}

/* Return the number of weights that do not match, where 0 means both are fully synchronised
*/
int compareWeights(struct NeuralNetwork neuralNet1, struct NeuralNetwork neuralNet2, int k, int n) {
    int weightsOf1[k*n];
    int weightsOf2[k*n];
    int sumOfDifference = 0;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            weightsOf1[(i * n) + j] = neuralNet1.weights[i][j];
            weightsOf2[(i * n) + j] = neuralNet2.weights[i][j];
        }
    }

    for (int i = 0; i < k * n; i++) {
        if (weightsOf1[i] != weightsOf2[i]) {
            sumOfDifference++;
        }
    }

    return sumOfDifference;
}

/** Prints the weights of each of the neural nets in the program in order of rank
*/
void printInitialWeights(int comm_sz,
                         int myRank,
                         int k,
                         int n,
                         struct NeuralNetwork neuralNetA,
                         struct NeuralNetwork neuralNetB,
                         struct NeuralNetwork my_neuralNetC) {
    for(int i = 0; i < comm_sz; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == myRank) {
            printf("\n==== BEFORE PROTOCOL RUN in Process %i======\n", myRank);
            printf("\n========== Network A ==========\n");
            printNetworkWeights(neuralNetA, k, n);
            printf("\n========== Network B ==========\n");
            printNetworkWeights(neuralNetB, k, n);
            printf("\n========== Network C ==========\n");
            printNetworkWeights(my_neuralNetC, k, n);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

void printFinal(char outputFlag[3],
                bool status,
                int comm_sz,
                int myRank,
                int epoch,
                int EPOCH_LIMIT,
                int synchThreshold,
                int my_synchTimeAB,
                int my_synchTimeAC,
                int k,
                int n,
                int l,
                int sumOfDifferenceAB,
                int sumOfDifferenceAC,
                int sumOfDifferenceBC,
                struct NeuralNetwork neuralNetA,
                struct NeuralNetwork neuralNetB,
                struct NeuralNetwork my_neuralNetC,
                double wallTime) {
    if (strcmp(outputFlag, "-v") == 0) {
        for(int i = 0; i < comm_sz; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == myRank) {
                printf("\n==== AFTER PROTOCOL RUN from process %i =======\n", myRank);
                if (status == true) {
                    printf("Network synchronised after %d total epochs\n", epoch);
                    printf("Network A and B synchronised for %d cycles\n", my_synchTimeAB);
                    printf("Network A and C synchronised for %d cycles\n", my_synchTimeAC);
                    printf("\n========== Network A ==========\n");
                    printNetworkWeights(neuralNetA, k, n);
                    printf("\n========== Network B ==========\n");
                    printNetworkWeights(neuralNetB, k, n);
                    printf("\n========== Network C ==========\n");
                    printNetworkWeights(my_neuralNetC, k, n);
                    printf("\n======== Comparison Score ======\n");
                    printf("A and B have %i differences in weights\n", sumOfDifferenceAB);
                    printf("A and C have %i differences in weights\n", sumOfDifferenceAC);
                    printf("B and C have %i differences in weights\n", sumOfDifferenceBC);

                } else {
                    printf("Network did not synchronise after %d epochs\n", epoch);
                    printf("\n========== Network A ==========\n");
                    printNetworkWeights(neuralNetA, k, n);
                    printf("\n========== Network B ==========\n");
                    printNetworkWeights(neuralNetB, k, n);
                    printf("\n========== Network C ==========\n");
                    printNetworkWeights(my_neuralNetC, k, n);
                    printf("Networks are unsynchronised after %d epochs.", EPOCH_LIMIT);
                    printf("\n======== Comparison Score ======\n");
                    printf("A and B have %i differences in weights\n", sumOfDifferenceAB);
                    printf("A and C have %i differences in weights\n", sumOfDifferenceAC);
                    printf("B and C have %i differences in weights\n", sumOfDifferenceBC);
                }
            }
        }
    } else if (strcmp(outputFlag, "-s") == 0) {
        for(int i = 0; i < comm_sz; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == myRank) {
                if (status == true) {
                    printf("In Process %d Network A and C synched after %i cycles and A and B after %i in a total number of %i epochs\n", myRank, my_synchTimeAC, my_synchTimeAB, epoch);
                } else {
                    printf("In Process %d Network A and C synched after %i cycles and A and B did not synch in a total number of %i epochs\n", myRank, my_synchTimeAC, epoch);
                }
            }
        }
    } else if (strcmp(outputFlag, "-p") == 0) {
        for(int i = 0; i < comm_sz; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == myRank) {
                printf("{'process': %d, 'commSz': %i, 'ACsynch': %i, 'ABsynch': %i, 'epochs': %i, 'synchThreshold': %i, 'k': %i, 'n': %i, 'l': %i, 'sumDiffAB': %i, 'sumDiffAC': %i 'sumDiffBC':%i, 'timeTaken':%lf}\n", myRank, comm_sz, my_synchTimeAC, my_synchTimeAB, epoch, synchThreshold, k, n, l, sumOfDifferenceAB, sumOfDifferenceAC, sumOfDifferenceBC, wallTime);
            }
        }
    } else if (strcmp(outputFlag, "-c") == 0) {
        for(int i = 0; i < comm_sz; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == myRank) {
                printf("%d, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %lf\n", myRank, comm_sz, my_synchTimeAC, my_synchTimeAB, epoch, synchThreshold, k, n, l, sumOfDifferenceAB, sumOfDifferenceAC, sumOfDifferenceBC, wallTime);
            }
        }
    }
}
