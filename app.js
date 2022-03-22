var Neuron = /** @class */ (function () {
    function Neuron(inputs, output) {
        this.alpha = 1;
        this.inputs = inputs;
        this.output = output;
        this.resetNetwork(this.weights);
    }
    Neuron.prototype.getWeightSum = function (inputs) {
        return this.weights.reduce(function (a, b, index) { return a + b * inputs[index]; });
    };
    Neuron.prototype.resetNetwork = function (weights) {
        this.inputs.forEach(function () { return weights.push(Math.random()); });
    };
    Neuron.prototype.activate = function () {
        return 1 / (1 + Math.exp(-this.alpha * this.getWeightSum(this.inputs)));
    };
    return Neuron;
}());
var Perceptron = /** @class */ (function () {
    function Perceptron(layersCount, neuronsCount, inputs) {
        this.neurons = new Array();
        this.layersCount = layersCount;
        this.neuronsCount = neuronsCount;
        this.inputs = inputs;
    }
    return Perceptron;
}());
console.log('hello');
