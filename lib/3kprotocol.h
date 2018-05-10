/**
 *
 * C Library for:
 * 1. 3K or KKK Protocol simulation, and
 * 2. Simulation of geometric and genetic security attacks on the protocol.
 * Author: Maduka Attamah
 *
 * Copyright 2016-2017 Maduka Attamah
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/* Struct type for the neural network */
struct NeuralNetwork {
    int** weights;
    int* hiddenLayerOutputs;
    int networkOutput;
} neuralNet;

/* Custom boolean type */
typedef enum {
    true, false
} bool;

/* Function declarations */
bool runKKKProtocol(struct NeuralNetwork, struct NeuralNetwork, int**, int, int, int, int syncThreshold, int epochLimit);
bool runGeometricAttackKKKProtocol(struct NeuralNetwork, struct NeuralNetwork, struct NeuralNetwork, int**, int, int, int, int syncThreshold, int epochLimit, int* epochFinal);
bool runGeneticAttackKKKProtocol(struct NeuralNetwork, struct NeuralNetwork, struct NeuralNetwork*, int**, int, int, int, int, int syncThreshold, int epochLimit, int* epochFinal);

struct NeuralNetwork constructNeuralNetwork(int, int, int);
struct NeuralNetwork cloneNeuralNetwork(int, int, struct NeuralNetwork);

int weightRand(int);
void initWeights(int**, int, int, int);
void updateWeights(struct NeuralNetwork, int**, int, int, int);
void updateWeightsGivenHLOutputs(struct NeuralNetwork, int**, int, int, int, int*);
void printNetworkWeights(struct NeuralNetwork, int, int);

int binaryRand(void);
int** getRandomInputs(int, int);
int getMinInputSumNeuron(struct NeuralNetwork, int**, int, int);

int* getHiddenLayerOutputs(struct NeuralNetwork, int**, int, int);
int getNetworkOutput(struct NeuralNetwork, int**, int, int);

void freeMemoryForNetwork(struct NeuralNetwork, int, int);

int** binaryToHLOutputs(int, int);
int* binaryCombinations(int, int);
int countZeros(int);


/**
 * Simulates the 3k protocol between two networks A and B. After the simulation the network weights for both network can be printed to show they are
 * synchronised. Use the utility function printNetworkWeights(...) in this library to print the network weights of network A and network B and attacker network.
 *
 * @param neuralNetA - neuralNetA and neuralNetB are the normal communicating pair by which we wish to generate a common key.
 * @param neuralNetB
 * @param attackerNet - or neuralNetC which is for the attacker.
 * @param inputs - the kth 'row' of the 'two-dimensional' array contains the inputs to the kth neuron.
 * @param k - identifies the number of hidden neurons.
 * @param n - identifies the n of inputs into each hidden neurons. The total number of inputs to the network is therefore N = k*n.
 * @param l - is the bound (-l to l) on the range of values that can be assigned to the weights. It is proposed that the bigger the l, the more
 *            difficult it is to break the protocol.
 * @param syncThreshold - if the all the involved networks produce the same weights in 'syncThreshold' successive rounds,
 *                         then we take it that the synchronisation is now stable and we can take the weights as final.
 *
 * @param epochLimit  - in case the networks are taking too long to reach synchronisation stability, we set this limit on the number of rounds that
 *                  can be executed so that we don't run the simulation for ever. This limit will depend on the resources available to your simulation
 *                  environment.

 * @return true or false indicating whether synchronisation was reached or not.
 */
bool runKKKProtocol(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, int** inputs, int k, int n, int l, int syncThreshold, int epochLimit) {
    int s = 0;
    int epoch = 0;

    while ((s < syncThreshold) && (epoch < epochLimit)) {
        int outputA = getNetworkOutput(neuralNetA, inputs, k, n);
        int outputB = getNetworkOutput(neuralNetB, inputs, k, n);

        if (outputA == outputB) {
            s = s + 1;
            updateWeights(neuralNetA, inputs, k, n, l);
            updateWeights(neuralNetB, inputs, k, n, l);
        } else {
            s = 0;
        }

        free(inputs);
        inputs = getRandomInputs(k, n);

        epoch = epoch + 1;
    }

    if (s == syncThreshold) {
        return true;
    }
    return false;

}


/**
 * Simulates the geometric attack on the 3k protocol
 * @param neuralNetA - neuralNetA and neuralNetB are the normal communicating pair by which we wish to generate a common key.
 * @param neuralNetB
 * @param attackerNet - or neuralNetC which is for the attacker.
 * @param inputs - the kth 'row' of the 'two-dimensional' array contains the inputs to the kth neuron.
 * @param k - identifies the number of hidden neurons.
 * @param n - identifies the n of inputs into each hidden neurons. The total number of inputs to the network is therefore N = k*n.
 * @param l - is the bound (-l to l) on the range of values that can be assigned to the weights. It is proposed that the bigger the l, the more
 *            difficult it is to break the protocol.
 * @param syncThreshold - if the all the involved networks produce the same weights in 'syncThreshold' successive rounds,
 *                         then we take it that the synchronisation is now stable and we can take the weights as final.
 * @param epochLimit  - in case the networks are taking too long to reach synchronisation stability, we set this limit on the number of rounds that
 *                  can be executed so that we don't run the simulation for ever. This limit will depend on the resources available to your simulation
 *                  environment.
 * @param epochFinal - int pointer passed from the calling context to collect the final epoch reached.
 * @return true or false indicating whether synchronisation was reached or not. Synchronisation is reached when the attack succeeds i.e the attacker succeeds in synchronising its
 *          network weights with that of network A and network B.
 */
bool runGeometricAttackKKKProtocol(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork attackerNet, int** inputs, int k, int n, int l, int syncThreshold, int epochLimit, int* epochFinal) {
    int s = 0;
    int epoch = 0;

    while ((s < syncThreshold) && (epoch < epochLimit)) {
        int outputA = getNetworkOutput(neuralNetA, inputs, k, n);
        int outputB = getNetworkOutput(neuralNetB, inputs, k, n);
        int outputC = getNetworkOutput(attackerNet, inputs, k, n);

        if ((outputA == outputB) && (outputA == outputC)) {
            //Increase synchronisation count, s.
            s = s + 1;

            //Update the weights of all three networks using the anti-Hebbian learning rule.
            updateWeights(neuralNetA, inputs, k, n, l);
            updateWeights(neuralNetB, inputs, k, n, l);
            updateWeights(attackerNet, inputs, k, n, l);

        } else if ((outputA == outputB) && (outputA != outputC)) {
            //Reset the synchronisation count - there was no synchronisation or sychronisation broke down in the round.
            s = 0;

            // Get the hidden neuron in C (attackerNet) for which the sum of its weights * inputs is minimised.
            int kthHidden = getMinInputSumNeuron(attackerNet, inputs, k, n);

            //Negate the output of the "minimum sum neuron" obtained above.
            attackerNet.hiddenLayerOutputs[kthHidden] = attackerNet.hiddenLayerOutputs[kthHidden] * (-1);

            //Now update the weights of C with the new hidden "bits" (outputs).
            updateWeights(attackerNet, inputs, k, n, l);

            //Update A and B's weights too, using the  anti-Hebbian learning rule.
            updateWeights(neuralNetA, inputs, k, n, l);
            updateWeights(neuralNetB, inputs, k, n, l);

        } else {
            //Reset the synchronisation count - there was no synchronisation or sychronisation broke down in the round.
            s = 0;
        }
        //Prepare for next round.
        free(inputs);

        //Get new random inputs for the next round.
        inputs = getRandomInputs(k, n);

        //Increment the round count. We will not run the protocol for ever - we will stop after a predefined number of rounds if
        //synchronisation has not been reached by then.
        epoch = epoch + 1;
    }

    // Set the final epoch for use by the calling context.
    *epochFinal = epoch;

    //Did the above while loop stop because the synchronisation threshold was reached?
    if (s == syncThreshold) {
        return true;  // We have succesfully synchronised the network. The weights were the same for syncThreshold number of rounds!
    }

    return false; //We've exceeded the epoch limit without succeeding in synchronising the network.
}


/**
 * Simulates the genetic attack on the 3k protocol. After the simulation the network weights for both network can be printed to show they are
 * synchronised. Use the utility function printNetworkWeights(...) in this library to print the network weights of network A and network B and attacker network.
 *
 * @param neuralNetA - neuralNetA and neuralNetB are the normal communicating pair by which we wish to generate a common key.
 * @param neuralNetB
 * @param attackerNet - or neuralNetC which is for the attacker.
 * @param inputs - the kth 'row' of the 'two-dimensional' array contains the inputs to the kth neuron.
 * @param k - identifies the number of hidden neurons.
 * @param n - identifies the n of inputs into each hidden neurons. The total number of inputs to the network is therefore N = k*n.
 * @param l - is the bound (-l to l) on the range of values that can be assigned to the weights. It is proposed that the bigger the l, the more
 *            difficult it is to break the protocol.
 * @param syncThreshold - if the all the involved networks produce the same weights in 'syncThreshold' successive rounds,
 *                         then we take it that the synchronisation is now stable and we can take the weights as final. For the genetic attack,
 *                         this value would typically be a small positive integer. Since we count a synchronisation as when all the networks in each
 *                         population have the same weights, a 'syncThreshold' of 1 would resemble a geometric attack setting, in the sense that the
 *                         the multiple populations will be comparable to the multiple successive synchronised rounds that we have in a geometric attack setting.
 * @param epochLimit  - in case the networks are taking too long to reach synchronisation stability, we set this limit on the number of rounds that
 *                  can be executed so that we don't run the simulation for ever. This limit will depend on the resources available to your simulation
 *                  environment.
 * @param epochFinal - int pointer passed from the calling context to collect the final epoch reached.
 * @return true or false indicating whether synchronisation was reached or not. Synchronisation is reached when the attack succeeds i.e the attacker succeeds in synchronising its
 *          network weights with that of network A and network B.
 */
bool runGeneticAttackKKKProtocol(struct NeuralNetwork neuralNetA, struct NeuralNetwork neuralNetB, struct NeuralNetwork* attackerNets, int** inputs, int k, int n, int l, int m, int syncThreshold, int epochLimit, int* epochFinal) {
    int s = 0;
    int epoch = 0;

    while ((s < syncThreshold) && (epoch < epochLimit)) {
        int outputA = getNetworkOutput(neuralNetA, inputs, k, n);
        int outputB = getNetworkOutput(neuralNetB, inputs, k, n);

        if ((outputA == outputB) && (sizeof (attackerNets) <= m)) {
            struct NeuralNetwork* newAttackerNets = malloc(sizeof (struct NeuralNetwork*) * sizeof (attackerNets) * pow(2, k - 1));
            int index = 0;
            //Get all the combinations that give us the outputA (or outputB).
            int** hlOutputs = binaryToHLOutputs(k, outputA);
            struct NeuralNetwork nn;
            for (int i = 0; i < sizeof (attackerNets); i++) {
                for (int j = 0; j < sizeof (hlOutputs); j++) {
                    //Clone attackerNet[i]
                    nn = cloneNeuralNetwork(k, n, attackerNets[i]);
                    //Update the weights of the new clone using given hidden layer outputs
                    updateWeightsGivenHLOutputs(nn, inputs, k, n, l, hlOutputs[j]);
                    //Add the new clone to newAttackerNets
                    newAttackerNets[index] = nn;
                    index++;
                }
            }
            //Update original attackerNets
            free(attackerNets);
            attackerNets = newAttackerNets;
            //Free memory for the temporary attacker network
            freeMemoryForNetwork(nn, k, n);

        } else if ((outputA == outputB) && (sizeof (attackerNets) > m)) {
            //Delete all the networks in the population whose outputs don't agree with that of  A and B; update the weights of the rest
            struct NeuralNetwork* newAttackerNets = malloc(sizeof (struct NeuralNetwork*) * m);
            int index = 0;
            for (int i = 0; i < sizeof (attackerNets); i++) {
                if (index >= m) break; //Caps the pool at m.
                if (outputA = getNetworkOutput(attackerNets[i], inputs, k, n)) {
                    //Update the weights as usual
                    updateWeights(attackerNets[i], inputs, k, n, l);
                    newAttackerNets[index] = attackerNets[i];
                    index++;
                }
            }
            if (index == m) {
                s++; //count the number of times we have all networks in the pool having the same output as A and B.
            }
            //Update original attackerNets
            free(attackerNets);
            attackerNets = newAttackerNets;

        } else {
            s = 0;
        }
        free(inputs);
        inputs = getRandomInputs(k, n);

        epoch = epoch + 1;
    }

    // Set the final epoch for use by the calling context.
    *epochFinal = epoch;

    //Did the simulation end due to reach of synchronisation threshold?
    if (s == syncThreshold) {
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
struct NeuralNetwork constructNeuralNetwork(int k, int n, int l) {
    struct NeuralNetwork neuralNetwork;
    // Allocate memory block for the neural network weights;
    neuralNetwork.weights = malloc(sizeof (int*) * (k));

    for (int i = 0; i < k; i++) {
        neuralNetwork.weights[i] = malloc(sizeof (int) * n);
    }

    //Allocate memory blocks for the hidden layer outputs.
    neuralNetwork.hiddenLayerOutputs = malloc(sizeof (int) * k);
    initWeights(neuralNetwork.weights, k, n, l);

    return neuralNetwork;
}

/**
 * Clone a new neural network using the one we supply, maintaining the structure and parameters of the supplied neural network.
 * @param k The number of perceptrons in the network to be cloned.
 * @param n The number of inputs per perceptron in the network to be cloned.
 * @param neuralNet The network to be cloned.
 * @return The clone of the supplied network.
 */
struct NeuralNetwork cloneNeuralNetwork(int k, int n, struct NeuralNetwork neuralNet) {
    struct NeuralNetwork neuralNetwork;
    // Allocate memory block for the neural network;
    neuralNetwork.weights = malloc(sizeof (int*) * (k));
    for (int i = 0; i < k; i++) {
        neuralNetwork.weights[i] = malloc(sizeof (int) * n);
    }
    neuralNetwork.hiddenLayerOutputs = malloc(sizeof (int) * k);

    //Initialise the weights
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            neuralNetwork.weights[i][j] = neuralNet.weights[i][j];
        }
    }

    //Initialise the hidden layer outputs
    for (int i = 0; i < k; i++) {
        neuralNetwork.hiddenLayerOutputs[i] = neuralNet.hiddenLayerOutputs[i];
    }

    return neuralNetwork;

}

/**
 * Gets the neuron/perceptron whose sum of product of inputs and weights is the minimum, of all the perceptrons in the network.
 * @param neuralNetwork The network to be processed.
 * @param inputs to the network (not part of the NeuralNetwork structure).
 * @param k The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron.
 * @return  The index of the minimum input sum neuron.
 */
int getMinInputSumNeuron(struct NeuralNetwork neuralNetwork, int** inputs, int k, int n) {
    int sum = 0;
    int minSum = 0;
    int minSumNeuron = 0;

    // Calculate the sum of product of inputs and weights for each perceptron, and
    // keep track of the minimum of all the perceptrons.
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            sum = sum + (inputs[i][j] * neuralNetwork.weights[i][j]);
        }
        //To get absolute value
        sum = abs(sum);
        // If current sum of product of inputs and weights is more than our previous
        // minimum, then we've got a new minimum.
        if ((minSum == 0) || (sum < minSum)) { //For the initial
            minSum = sum;
            minSumNeuron = i;
        }
        sum = 0;  // Ready for next perceptron.
    }

    return minSumNeuron;
}

/**
 * Generates a random weight whose value is between -l and l inclusive.
 * @param l
 * @return The generated random weight.
 */
int weightRand(int l) {
    int randomNum = rand() % (2 * l + 1); // Now we have 0 to 2l.
    return randomNum - l;  // Now we have -l to l.
}

/**
 * Get random weights (-l to l) for a neural network with k perceptrons and n inputs per perceptron.
 * @param weights
 * @param k
 * @param n
 * @param l
 */
void initWeights(int** weights, int k, int n, int l) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            weights[i][j] = weightRand(l);
        }
    }
}


/**
 * Updates the weight vectors of a network using the anti-Hebbian learning rule: w(i) = w(i) - output * input(i)
 * @param neuralNet The network whose weight is to be updated.
 * @param inputs The input vector containing the inputs to the network.
 * @param k  The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron in the network.
 * @param l  The upperbound (l) and lower bound (-l) of weight to be assigned.
 *
 */
void updateWeights(struct NeuralNetwork neuralNet, int** inputs, int k, int n, int l) {
    int* hlOutputs = getHiddenLayerOutputs(neuralNet, inputs, k, n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            // Update the weight using anti-Hebbian learning rule.
            neuralNet.weights[i][j] = neuralNet.weights[i][j] - (hlOutputs[i] * inputs[i][j]);

            //Ensure that the new weight is not above l and not below -l
            if (neuralNet.weights[i][j] < ((-1) * l)) {
                neuralNet.weights[i][j] = (-1) * l;
            } else if (neuralNet.weights[i][j] > l) {
                neuralNet.weights[i][j] = l;
            }
        }
    }
}


/**
 * Works like function updateWeights(...). The difference here is that the hidden layer outputs are not generated by the network, but
 * supplied externally. This is useful for the protocol attack simulation because sometimes the attacker needs to artificially set it's
 * hidden layer output vector, and generate new weights based on this artificially-set hidden layer output vector.
 *
 * @param neuralNet The network whose weight is to be updated.
 * @param inputs The input vector containing the inputs to the network.
 * @param k  The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron in the network.
 * @param l  The upperbound (l) and lower bound (-l) of weight to be assigned.
 * @param hlOutputs The supplied hidden layer outputs.
 */
void updateWeightsGivenHLOutputs(struct NeuralNetwork neuralNet, int** inputs, int k, int n, int l, int* hlOutputs) {
    //int* hlOutputs = getHiddenLayerOutputs(neuralNet, inputs, k, n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            neuralNet.weights[i][j] = neuralNet.weights[i][j] - (hlOutputs[i] * inputs[i][j]);
            if (neuralNet.weights[i][j] < ((-1) * l)) {
                neuralNet.weights[i][j] = (-1) * l;
            } else if (neuralNet.weights[i][j] > l) {
                neuralNet.weights[i][j] = l;
            }
        }
    }
}

/**
 * Prints the weights of a neural network.
 * @param neuralNet The network whose weight is to be printed.
 * @param k The number of perceptrons in the neural network.
 * @param n  The number of inputs to each perceptron in the neural network.
 */
void printNetworkWeights(struct NeuralNetwork neuralNet, int k, int n) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d, ", neuralNet.weights[i][j]);
        }
        printf("\n");
    }
}

/**
 * Generates a random number from the set {-1, 1}.
 * @return The generated random number.
 */
int binaryRand() {
    int randNum = rand();
    if (randNum % 2 == 0) { // Even number
        return 1;
    } else {                // Odd number
        return -1;
    }
}

/**
 * Generates random inputs value (each input value is either -1 or 1), to be used for a neural
 * network with k perceptrons and n inputs per perceptron.
 * @param k
 * @param n
 * @return The input vector generated.
 */
int** getRandomInputs(int k, int n) {

    //Allocate memory block for the inputs
    int** inputs = malloc(sizeof (int*) * k);
    for (int i = 0; i < k; i++) {
        inputs[i] = malloc(sizeof (int) * n);
    }

    // Now generate and set the inputs.
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            inputs[i][j] = binaryRand();
        }
    }
    return inputs;
}

/**
 * Trigger the hidden layer outputs for the supplied neural network, and then return the hidden
 * layer output vector.
 * @param neuralNet The network whose hidden layer outputs is to be triggered.
 * @param inputs  The inputs to the network.
 * @param k  The number of perceptrons in the network.
 * @param n  The number of inputs to each perceptron.
 * @return   The hidden layer outputs of the supplied network.
 */
int* getHiddenLayerOutputs(struct NeuralNetwork neuralNet, int** inputs, int k, int n) {
    int sum = 0;

    //Allocate memory for the hidden layer output vector.
    int* hlOutputs = malloc(sizeof (int) * k);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            sum = sum + (neuralNet.weights[i][j] * inputs[i][j]);
        }
        //Each hidden layer output must be either -1 or +1. We are interested in
        // only the sign parity(negative or positive) of the output of each perceptron.
        if (sum <= 0) {
            hlOutputs[i] = -1;
        } else {
            hlOutputs[i] = 1;
        }
    }
    return hlOutputs;
}

/**
 * Trigger the output of the neural network and return it.
 * @param neuralNet The network whose output is to be obtained.
 * @param inputs  The inputs to the network.
 * @param k  The number of perceptrons to the network.
 * @param n The number of inputs to each perceptron in the network.
 * @return The value of the output of the network.
 *
 */
int getNetworkOutput(struct NeuralNetwork neuralNet, int** inputs, int k, int n) {
    int* hlOutputs = getHiddenLayerOutputs(neuralNet, inputs, k, n);

    //Obtain the product of all the hidden layer outputs. Since each hidden layer
    //output is either 1 or -1, this product will give us a sign parity (positive or negative).
    int prod = 1;
    for (int i = 0; i < k; i++) {
        prod = prod * (hlOutputs[i]);
    }

    return prod;
}

/**
 * Free up the memory allocated for a neural network.
 * @param neuralNet
 * @param k The number of perceptrons in the neural network.
 * @param n  The number of inputs to each perceptron in the network.
 */
void freeMemoryForNetwork(struct NeuralNetwork neuralNet, int k, int n) {
    // Free memory block for the weight vectors of the neural network;
    for (int i = 0; i < k; i++) {
        free(neuralNet.weights[i]);
    }
    free(neuralNet.weights);

    //Free memory for the hidden layer outputs.
    free(neuralNet.hiddenLayerOutputs);
}


/**
 * Gets all the hidden layer output vectors whose products gives rise to "output". This is a utility function we use in the genetic attack of the 3k protocol.
 * For example, suppose we have k=2 perceptrons, and the final output
 * of the network is -1. All the possible hidden layer output vectors are: (-1,-1), (-1,1), (1,-1), and (1,1): (2^k - 1 in all). However the valid ones are only
 * those whose products produced the parity of the final output. In this case one valid hidden layer output vectors would be (-1,1) and (1,-1) (For example,
 * for (-1,1), -1 * 1 = -1 (hidden layer output from perceptron one = -1 and hidden layer output from perceptron two = 1; therefore final output of network is -1
 * and this is equal to the given output)).
 *
 * @param k  The number of perceptrons in the neural network.
 * @param output The final output of the neural network for which we would like to obtain valid hidden layer output vectors.
 * @return  This function returns the desired vectors.
 */
int** binaryToHLOutputs(int k, int output) {
    int* bCombinations = binaryCombinations(k, output);
    //Allocate memory space for the output
    int** hlOutputs = malloc(sizeof (bCombinations) * sizeof (int*));
    for (int i = 0; i < sizeof (bCombinations); i++) {
        hlOutputs[i] = malloc(sizeof (int) * k);
    }
    //Iterate through the bits of each number in bCombinations
    for (int i = 0; i < sizeof (bCombinations); i++) {
        for (int j = 0; j < k; j++) {
            if ((bCombinations[i] & 1) == 1) {
                hlOutputs[i][j] = 1;
            } else {
                hlOutputs[i][j] = -1;
            }
            bCombinations[i] = bCombinations[i] >> 1;
        }
    }
    return hlOutputs;
}


/**
 * Get all the binary numbers for which the product of its bits (when we substitute -1 for 0, and 1 for 1) gives rise to "output". This is
 * a utility function we use in the genetic attack of the 3k protocol. For example, suppose we have k=2 perceptrons, and the final output
 * of the network is -1. All the possible hidden layer outputs are: 00, 01, 10, and 11 (2^k - 1 in all). If we substitute -1 for 0 and 1 for 1
 * in each of these binary combinations, then we can get the combinations that produced the parity of the final output. In this case one valid combination
 * will be 01 (-1 * 1 = -1 (hidden layer output from perceptron one = -1 and hidden layer output from perceptron two = 1; therefore final output of network is -1
 * and this is equal to the given output));  another valid hidden layer output combination for the given final output is 10 (1 and -1).
 *
 * @param k  The number of perceptrons in the neural network.
 * @param output The final output of the neural network for which we would like to obtain valid hidden layer output vectors.
 * @return  This function however returns the desired vector using binary digits (0,1). The function binaryToHLOutputs(...) will convert these binary
 * digits to -1 and 1 (-1 for zero, and 1 for one), giving the actual hidden layer outputs.
 */
int* binaryCombinations(int k, int output) {
    int* combinations = malloc(sizeof (int) * (int) pow(2.0, k - 1));  // Only half of the all the possible binary combinations will be valid (i.e. produce -ve, or +ve parity)
    int index = 0;
    if (output == -1) {  // Look for binary combinations that produce -ve parity.
        for (int i = 0; i < (int) pow(2.0, k); i++) {
            if (countZeros(i) % 2 == 1) { // The number of zeros in i is odd i.e odd number of -1s, which will give output parity of -1
                combinations[index] = i;
                index++;
            }
        }
    } else if (output == 1) {  // Look for binary combinations that produce +ve parity.
        for (int i = 0; i < (int) pow(2.0, k); i++) {
            if (countZeros(i) % 2 == 0) { // The number of zeros in i is even i.e even number of -1s, which will give output parity of +1
                combinations[index] = i;
                index++;
            }
        }
    }
    return combinations;
}

/**
 * Utility function: counts the number of zero bits in a number.
 */
int countZeros(int k) {
    int c = 0;
    int x = 0;
    while (k != 0) {
        x++;
        if ((k & 1) == 1) {
            c++;
        }
        k = k >> 1;
    }
    return (x - c);
}
