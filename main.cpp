#include <iostream>
#include <math.h>

#include "randlib.h" 
#include "mnist/mnist.h"

using namespace std;

#define numOfInputNodes 785
#define numOfFirstHiddenLayerNodes 10
#define numOfSecondHiddenLayerNodes 10
#define numOfOutputNodes 2

void randomizeWeightMatrixForFirstHidden(float weights[numOfFirstHiddenLayerNodes][numOfInputNodes]) {
    for(int i = 0; i < numOfFirstHiddenLayerNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            weights[i][j] = rand_weight();
        }
    }
}

void randomizeWeightMatrixForSecondHidden(float weights[numOfSecondHiddenLayerNodes][numOfFirstHiddenLayerNodes]) {
    for(int i = 0; i < numOfSecondHiddenLayerNodes; i++) {
        for(int j = 0; j < numOfFirstHiddenLayerNodes; j++) {
            weights[i][j] = rand_weight();
        }
    }
}


void randomizeWeightMatrixForOutPut(float weights[numOfOutputNodes][numOfSecondHiddenLayerNodes]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfSecondHiddenLayerNodes; j++) {
            weights[i][j] = rand_weight();
        }
    }
}

bool isPrime(int number) {
    if (number < 2)
        return false;
    if (number == 2)
        return true;
    if (number % 2 == 0)
        return false;
    for (int i = 3; (i * i) <= number; i += 2) {
        if(number % i == 0 )
            return false;
    }
    return true;
}

void initTarget(float target[], int numberOnPicture) {
    target[0] = ((numberOnPicture % 2) == 0) ? 0 : 1; // odd or even number
    target[1] = isPrime(numberOnPicture) ? 0 : 1;
}



void get_output_first_hidden(float hiddenLyaer[], int input[], float weights[numOfFirstHiddenLayerNodes][numOfInputNodes]) {
    
    for(int i = 0; i < numOfFirstHiddenLayerNodes; i++) {
        float resultOfMultiplication = 0;
        for(int j = 0; j < numOfInputNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j];
        }
        hiddenLyaer[i] = resultOfMultiplication;
    }
    hiddenLyaer[numOfFirstHiddenLayerNodes - 1] = 1; //bias for hidden nodes
}

void get_output_second_hidden(float secondHiddenLayerNodes[], float firstHiddenLayerNodes[], float weights[numOfSecondHiddenLayerNodes][numOfFirstHiddenLayerNodes]) {
    
    for(int i = 0; i < numOfSecondHiddenLayerNodes; i++) {
        float resultOfMultiplication = 0;
        for(int j = 0; j < numOfFirstHiddenLayerNodes; j++) {
            resultOfMultiplication += firstHiddenLayerNodes[j] * weights[i][j];
        }
        secondHiddenLayerNodes[i] = resultOfMultiplication;
    }
    secondHiddenLayerNodes[numOfSecondHiddenLayerNodes - 1] = 1; //bias for hidden nodes
}


void get_output(float output[], float input[], float weights[numOfOutputNodes][numOfSecondHiddenLayerNodes]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        float resultOfMultiplication = 0;
        for(int j = 0; j < numOfSecondHiddenLayerNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j];
        }
        output[i] = resultOfMultiplication;
    }
}

void squash_output(float output[]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
        // printf("squashed output[%d] = %f\n", i, output[i]);
    }
}

void squash_fist_hidden(float output[]) {
    
    for(int i = 0; i < numOfFirstHiddenLayerNodes; i++) {
        output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
        // printf("squashed output[%d] = %f\n", i, output[i]);
    }
}

void squash_second_hidden(float output[]) {
    
    for(int i = 0; i < numOfSecondHiddenLayerNodes; i++) {
        output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
        // printf("squashed output[%d] = %f\n", i, output[i]);
    }
}

void get_error_for_output(float errors[], float target[], float output[]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        errors[i] = (target[i] - output[i]) * output[i] * (1 - output[i]);
    }
}

void get_error_for_second_hidden_layer(float errorsOutput[], float errorsHidden[], float hiddenOutput[], float weights[numOfOutputNodes][numOfSecondHiddenLayerNodes]) {
	float resultOfMultiplication = 0;
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfSecondHiddenLayerNodes; j++) {
            resultOfMultiplication += errorsOutput[i] * weights[i][j];
        }
    }
    
    for(int i = 0; i < numOfSecondHiddenLayerNodes; i++) {
        errorsHidden[i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * resultOfMultiplication;
    }


}

void get_error_for_first_hidden_layer(float errorsSecondHidden[], float errorsFirstHidden[], float hiddenOutput[], float weights[numOfSecondHiddenLayerNodes][numOfFirstHiddenLayerNodes]) {
	float resultOfMultiplication = 0;
    for(int i = 0; i < numOfSecondHiddenLayerNodes; i++) {
        for(int j = 0; j < numOfFirstHiddenLayerNodes; j++) {
            resultOfMultiplication += errorsSecondHidden[i] * weights[i][j];
        }
    }
    
    for(int i = 0; i < numOfFirstHiddenLayerNodes; i++) {
        errorsFirstHidden[i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * resultOfMultiplication;
    }
}

float getAverageError(float error[]) {
    float errorsSum = 0;
    for (int i = 0; i < numOfOutputNodes; i++) {
        errorsSum += fabs(error[i]);
    }
    
    return (errorsSum / numOfOutputNodes);
}

void update_weights_output(float learningRate, float hiddenSecond[], float errors[], float weights[numOfOutputNodes][numOfSecondHiddenLayerNodes]) {
    float deltaWeights[numOfOutputNodes][numOfSecondHiddenLayerNodes];
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfSecondHiddenLayerNodes; j++) {
            deltaWeights[i][j] = learningRate  * hiddenSecond[j] * errors[i];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}


void update_weights_second_hidden(float learningRate, float hiddenfirst[], float errors[], float weights[numOfSecondHiddenLayerNodes][numOfFirstHiddenLayerNodes]) {
    float deltaWeights[numOfSecondHiddenLayerNodes][numOfFirstHiddenLayerNodes];
    for(int i = 0; i < numOfSecondHiddenLayerNodes; i++) {
        for(int j = 0; j < numOfFirstHiddenLayerNodes; j++) {
            deltaWeights[i][j] = learningRate * hiddenfirst[j] * errors[i];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}

void update_weights_first_hidden(float learningRate, int input[], float errors[], float weights[numOfFirstHiddenLayerNodes][numOfInputNodes]) {
    float deltaWeights[numOfFirstHiddenLayerNodes][numOfInputNodes];
    for(int i = 0; i < numOfFirstHiddenLayerNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            deltaWeights[i][j] = learningRate * input[j] * errors[i];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}

int main(int argc, char const *argv[]) {
    // --- an example for working with random numbers
    seed_randoms();
    
    float sampNoise = 0;
    
    // --- a simple example of how to set params from the command line
    if(argc == 2){ // if an argument is provided, it is SampleNoise
        sampNoise = atof(argv[1]);
        if (sampNoise < 0 || sampNoise > .5){
            printf("Error: sample noise should be between 0.0 and 0.5\n");
            return 0;
        }
    }
    
    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData;  // depends on loadType
    int loadType = 1; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    mnist_data *zTestingData;      // each image is 28x28 pixels
    unsigned int sizeTestingData;  // depends on loadType
    int loadTypeTesing = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zTestingData, &sizeTestingData, loadTypeTesing)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    float learningRate = 0.01;
    
    int inputNodes[numOfInputNodes];
    float firstHiddenLayerNodes[numOfFirstHiddenLayerNodes];
    float secondHiddenLayerNodes[numOfSecondHiddenLayerNodes];
    float outputNodes[numOfOutputNodes];
    
    float errorsFirstHidden[numOfFirstHiddenLayerNodes];
    float errorsSecondHidden[numOfSecondHiddenLayerNodes];
    float errorsOutput[numOfOutputNodes];
    
    float target[numOfOutputNodes];
    
    float weightsFirstHidden[numOfFirstHiddenLayerNodes][numOfInputNodes];
    randomizeWeightMatrixForFirstHidden(weightsFirstHidden);
    
    float weightsSecondHidden[numOfSecondHiddenLayerNodes][numOfFirstHiddenLayerNodes];
    randomizeWeightMatrixForSecondHidden(weightsSecondHidden);
    
    float weightsOutput[numOfOutputNodes][numOfSecondHiddenLayerNodes];
    randomizeWeightMatrixForOutPut(weightsOutput);
    
    for(int epoch = 0; epoch < 20; epoch++) {
        
        
        float resultError = 0;
        
        for(int picIndex = 0; picIndex < sizeTestingData; picIndex++) {
            get_input(inputNodes, zTestingData, picIndex, 0);
            
            initTarget(target, zTestingData[picIndex].label);
            
            get_output_first_hidden(firstHiddenLayerNodes, inputNodes, weightsFirstHidden);
            squash_fist_hidden(firstHiddenLayerNodes);
            
            get_output_second_hidden(secondHiddenLayerNodes, firstHiddenLayerNodes, weightsSecondHidden);
            squash_second_hidden(secondHiddenLayerNodes);
            
            get_output(outputNodes, secondHiddenLayerNodes, weightsOutput);
            squash_output(outputNodes);
            
            get_error_for_output(errorsOutput, target, outputNodes);
            resultError += getAverageError(errorsOutput);
            
        }
        
        cout << (resultError / sizeTestingData) << ", ";
        
        for(int picIndex = 0; picIndex < sizeData; picIndex++) {
            
           	get_input(inputNodes, zData, picIndex, 0);
            
            initTarget(target, zData[picIndex].label);
            
            get_output_first_hidden(firstHiddenLayerNodes, inputNodes, weightsFirstHidden);
            squash_fist_hidden(firstHiddenLayerNodes);
            
            get_output_second_hidden(secondHiddenLayerNodes, firstHiddenLayerNodes, weightsSecondHidden);
            squash_second_hidden(secondHiddenLayerNodes);
            
            get_output(outputNodes, secondHiddenLayerNodes, weightsOutput);
            squash_output(outputNodes);
            
            
            get_error_for_output(errorsOutput, target, outputNodes);
            update_weights_output(learningRate, secondHiddenLayerNodes, errorsOutput, weightsOutput);
            
            get_error_for_second_hidden_layer(errorsOutput, errorsSecondHidden, secondHiddenLayerNodes, weightsOutput);
            update_weights_second_hidden(learningRate, firstHiddenLayerNodes, errorsSecondHidden, weightsSecondHidden);

            get_error_for_first_hidden_layer(errorsSecondHidden, errorsFirstHidden, firstHiddenLayerNodes, weightsSecondHidden);
            update_weights_first_hidden(learningRate, inputNodes, errorsFirstHidden, weightsFirstHidden);
            
        }

        cout << endl;
        
    }
    
    
    
    
    return 0;
}
