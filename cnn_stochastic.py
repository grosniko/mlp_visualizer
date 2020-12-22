
import random
import plainchart
from graph import createGraph
from warnings import filterwarnings
from activation_functions import leakyReLU, sigmoid, identity
import json

def cnn(neurons_per_layer_array, randWeightsLimits, output_bias):
    #randrange function only returns integers, so to get floats between i.e. 0 & 1 must multiply limits by 100 and then divide the returned value by 100
    randWeightsLimits[0] = randWeightsLimits[0] * 100
    randWeightsLimits[1] = randWeightsLimits[1] * 100

    layer_num = 0
    weights = []
    bias_per_hidden_layer = []
    for current_layer_neurons_count in neurons_per_layer_array:
        if len(neurons_per_layer_array) > layer_num+1:
            weights.append([])
            next_layer_neurons_count = neurons_per_layer_array[layer_num+1]
            num_connections = current_layer_neurons_count * next_layer_neurons_count
            for next_layer_neuron_index in range(0,next_layer_neurons_count):
                weights[layer_num].append([])
                for current_layer_neuron_index in range(0, current_layer_neurons_count):
                    weights[layer_num][next_layer_neuron_index].append(random.randrange(randWeightsLimits[0],randWeightsLimits[1])/100)

            bias_per_hidden_layer.append(random.randrange(randWeightsLimits[0],randWeightsLimits[1])/100)
            layer_num += 1

    #remove bias from output layer if wanted
    if not output_bias:
        bias_per_hidden_layer.pop(-1)

    # print("weights")
    # print(weights)
    # print("biases")
    # print(bias_per_hidden_layer)
    return weights, bias_per_hidden_layer

def ReLU(value, derivative = False):
    if derivative:
        result = 0
        if value >= 0:
            result = 1.0
    else:
        result = max(value, 0)
    return result

def squared_error_CF(output, target, derivative = False):
    if derivative:
        result = output - target
    else:
        result = 0.5 * (output-target)**2
    return result


def forwardSingleDataPoint(inputs, weights, biases, activation_function):
    #activation_function = globals()[activation_function]()
    all_layers_activation_values = []
    all_layers_activation_values.append(inputs)
    #values that help iterate
    weight_layer_index = 0
    for weight_layer in weights:
        #values that we need to calculate
        right_layer_activation_values = []
        for one_set_of_the_connections in weight_layer:
            #values that we need to calculate
            sum_product_for_connection_plus_bias = 0

            #the values that we will multiply by the weights and sum to bias
            inputs_from_layer_to_the_left = all_layers_activation_values[weight_layer_index]
            #multiply each input by the corresponding weight and sum the total
            for weight_x_input_index in range(0, len(one_set_of_the_connections)):
                sum_product_for_connection_plus_bias += inputs_from_layer_to_the_left[weight_x_input_index] * one_set_of_the_connections[weight_x_input_index]

            #add the bias if it exists
            if len(biases) > weight_layer_index:
                sum_product_for_connection_plus_bias += biases[weight_layer_index]
            activation_value_for_neuron_in_layer_to_the_right = activation_function(sum_product_for_connection_plus_bias)
            right_layer_activation_values.append(activation_value_for_neuron_in_layer_to_the_right)
        all_layers_activation_values.append(right_layer_activation_values)
        weight_layer_index +=1
    # print("activation values")
    # print(all_layers_activation_values)
    return all_layers_activation_values

def backwardSingleDataPoint(activation_values, targets, weights, activation_function, cost_function):

    squared_error = 0
    outputs = activation_values[-1]

    all_backwards_inputs = []
    dLoss_dPreactivatedOutput = []
    for output_x_target_index in range(0, len(outputs)):
        squared_error += cost_function(outputs[output_x_target_index], targets[output_x_target_index])
        #calculate impact of activated and preactivated outputs on error and initialize chain rule
                                    #MEGA IMPORTANT --> output minus target, NEVER target - output or else goes in wrong direction.
        dTarget_dActivatedOutput =  cost_function(outputs[output_x_target_index], targets[output_x_target_index], True)
        dActivatedOutput_dPreactivatedOutput = activation_function(outputs[output_x_target_index], True)

        #chain rule
        dLoss_dPreactivatedOutput.append(dTarget_dActivatedOutput * dActivatedOutput_dPreactivatedOutput)

    all_backwards_inputs.append(dLoss_dPreactivatedOutput)

    weight_gradients = []
    bias_gradients = []
    for activation_value_set_index in range(2, len(activation_values)+1):
        #multiply by -1 to work backwards
        activation_value_set_index *= -1

        #multiply to find weight gradients
        current_backwards_inputs = all_backwards_inputs[-1]
        weight_gradients_for_layer = []

        for input in current_backwards_inputs:
            weight_gradients_for_layer.append([])
            for value in activation_values[activation_value_set_index]:
                weight_gradients_for_layer[-1].append(input * value)
        #weight_gradients.insert(0,weight_gradients_for_layer)
        weight_gradients.insert(0, weight_gradients_for_layer)
        #dont do next step if next activation value is actually the original inputs
        if activation_value_set_index > -1 * len(activation_values):
            #now we caculate the backwards inputs to prepare the next iteration of this forloop\
            intermediate_backwards_inputs = []
            #calculate gradient in between neurons - which is simply the weight of connection - and multiply by inputs
            weight_layer = weights[activation_value_set_index+1]
            #create array that will hold the partial gradients
            partial_gradients_summed = [0] * len(activation_values[activation_value_set_index])
            for weight_set_x_input_index in range(0,len(weight_layer)):
                index = 0
                for weight in weight_layer[weight_set_x_input_index]:
                    partial_gradients_summed[index] += weight * current_backwards_inputs[weight_set_x_input_index]
                    index += 1
                #intermediate_backwards_inputs.append(sum_gradient_for_neuron)
            intermediate_backwards_inputs = partial_gradients_summed
            #deactivate and multiply to calculate next inputs
            deactivated = []
            for value in activation_values[activation_value_set_index]:
                deactivated.append(activation_function(value, True))

            next_backwards_inputs = []
            for deactivated_x_input_index in range(0, len(deactivated)):
                next_backwards_inputs.append(deactivated[deactivated_x_input_index] * intermediate_backwards_inputs[deactivated_x_input_index])

            all_backwards_inputs.append(next_backwards_inputs)

    #now get the biases which are going to simply be each backwardinput's hidden layer's sums (exclude input and output layers where biases aren't added)
    for layer_index in range(0, len(all_backwards_inputs)):
        bias_gradients.append(sum(all_backwards_inputs[layer_index]))

    # print("weight gradients")
    # print(weight_gradients)
    # print("bias_gradients")
    # print(bias_gradients)

    return squared_error, weight_gradients, bias_gradients

def update(weights, biases, weight_gradients, bias_gradients, learning_rate, bias_learning_rate):
    #update weights
    for weight_layer_index in range(0, len(weights)):
        for weight_set_index in range(0, len(weights[weight_layer_index])):
            for weight_index in range(0, len(weights[weight_layer_index][weight_set_index])):
                weights[weight_layer_index][weight_set_index][weight_index] -= weight_gradients[weight_layer_index][weight_set_index][weight_index] * learning_rate
    #update biases
    for bias_index in range(0, len(biases)):
        biases[bias_index] -= bias_gradients[bias_index] * bias_learning_rate

    # print("updated weights")
    # print(weights)
    # print("updated biases")
    # print(biases)

    return weights, biases

def train(data = [[1],[2],[3]],
          targets = [[3],[2],[1]],
          layers = [1, 2, 1],
          output_bias = False,
          randWeightsLimits = [0, 1],
          activation_function = ReLU,
          cost_function = squared_error_CF,
          epochs = 1,
          learning_rate = 0.01,
          bias_learning_rate = None,
          pauseEpoch = False,
          pauseIteration = False,
          viz =[True, True, False]):  #[plainchart, graph at end of training, graph at start of epoch or iteration]
    #flexibility to have different learning rates for bias
    if bias_learning_rate == None:
        bias_learning_rate = learning_rate

    #set up neural network architecture
    weights, biases = cnn(layers, randWeightsLimits, output_bias)
    RMSEs = []
    #run epochs
    print("Training " + activation_function.__name__ + " nn of layers " + str(layers))

    RMSE = 0
    for epoch in range(0, epochs):
        #restart error measurement
        squared_error_sum = 0

        #shuffle the data and corresponding targets
        c = list(zip(data, targets))
        random.shuffle(c)
        data, targets = zip(*c)

        #iterate over each data point
        print("Epoch " + str(epoch + 1) + "/" + str(epochs))

        for data_x_targets_index in range(0, len(data)):
            #collect data points
            input_set = data[data_x_targets_index]
            target_set = targets[data_x_targets_index]
            #forward prop, back prop and update weights
            activation_values = forwardSingleDataPoint(input_set, weights, biases, activation_function)

            squared_error, weight_gradients, bias_gradients = backwardSingleDataPoint(activation_values, target_set, weights, activation_function, cost_function)

            if epoch == 0 and data_x_targets_index == 0 and viz[2] or pauseIteration:
                graph_name = activation_function.__name__ + " - Ep #" + str(epoch) + ", It #" + str(data_x_targets_index) + " @ lr " + str(learning_rate)
                createGraph(graph_name, activation_values, weights, biases, ["0.5 * error^2", squared_error])

            if pauseIteration:
                print("weight gradients")
                print(weight_gradients)

                print("bias gradients")
                print(bias_gradients)

                input("Press Enter to start next iteration...")

            weights, biases = update(weights, biases, weight_gradients, bias_gradients, learning_rate, bias_learning_rate)

            #add to sum to calculate RMSE at end of epoch
            squared_error_sum += squared_error

        RMSE = (squared_error_sum / len(data))**(0.5)
        #print chart in terminal

        if epoch > 0:
            RMSEs.append(RMSE)
        print("RMSE :" + str(RMSE))

        if pauseEpoch:
            print("weights")
            print(weights)

            print("biases")
            print(biases)

            print("weight gradients")
            print(weight_gradients)

            print("bias gradients")
            print(bias_gradients)

            input("Press Enter to start next epoch...")

    #print RMSE Chart
    if epochs > 1 and viz[0]:
        chart = plainchart.PlainChart(RMSEs, style=plainchart.PlainChart.scatter)
        print(chart.render())
    #create nn graph
    graph_name = activation_function.__name__ + " - Epoch " + str(epochs) + ' @ lr ' + str(learning_rate)
    if viz[1]:
        print("Creating NN graph...")
        graph = createGraph(graph_name, activation_values, weights, biases, ["Epoch #" + str(epochs) + " RMSE", str(RMSE)])

    #save model
    model = {"weights":weights,
             "biases":biases,
             "layers": layers,
             "lr":learning_rate,
             "af": str(activation_function),
             "cf": str(cost_function),
             "RMSE": RMSE,
             "name": graph_name,
             "activation_values": activation_values,
             "epochs": epochs}

    #notify
    print("Completed training of " + activation_function.__name__ + " nn of layers " + str(layers))
    return model


#functinos to create data
def createData(lower, upper, points, number):
    data = []
    for integer in range(0, number):
        datapoint = []
        for point in range(0, points):
            datapoint.append(random.randint(lower,upper))
        data.append(datapoint)
    return data

def createTargets(data, function):
    targets  = []
    for datapoint in data:
        targets.append(function(datapoint))
    return targets

def linear(datapoint, slope=2, intercept=5):
    sum = 0
    for value in datapoint:
        sum += value * slope + 5
    return [sum]

def quadratic(value, intercept=0):
    return [value[0]**2 + intercept]

def saveModel(model, name=""):
    if name == "":
        name = model["name"]
    with open(name, 'w', encoding='utf-8') as outfile:
        json.dump(model, outfile, ensure_ascii=False, indent=4)
    print("NN model " + name + " saved!")

def drawModel(model):
    createGraph(model["name"],
                model["activation_values"],
                model["weights"],
                model["biases"],
                ["Epoch #" + model["epochs"] + " RMSE", model["RMSE"]])


random.seed(42)
#Train and testingvalue[0]*slope + intercept
filterwarnings("ignore")
data = createData(1, 10, 1, 1000)
targets = createTargets(data, quadratic)
#
model = train(data = data,
              targets = targets,
              layers =[1,4,4,1],
              output_bias = True,
              randWeightsLimits=[0,1],
              epochs = 100,
              learning_rate = 0.00001,
              activation_function = ReLU,
              cost_function = squared_error_CF,
              pauseEpoch = False,
              pauseIteration = False,
              viz=[False, True, False])

# print(forwardSingleDataPoint([10], model["weights"], model["biases"], model["af"])[-1])
# print(forwardSingleDataPoint([1], model["weights"], model["biases"], model["af"])[-1])
# print(forwardSingleDataPoint([3], model["weights"], model["biases"], model["af"])[-1])
# print(forwardSingleDataPoint([11], model["weights"], model["biases"], model["af"])[-1])

saveModel(model, name="quadratic_1_to_10.json")
