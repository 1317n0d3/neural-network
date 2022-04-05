var Neuron = /** @class */ (function () {
    function Neuron(inputs) {
        this.alpha = 1;
        this.inputs = inputs;
        this.resetNetwork(this.weights);
        this.output = this.activate();
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
    Neuron.prototype.getOutput = function () {
        return this.output;
    };
    return Neuron;
}());
var Perceptron = /** @class */ (function () {
    function Perceptron(layersCount, neuronsCount, inputs) {
        if (layersCount === void 0) { layersCount = 3; }
        if (neuronsCount === void 0) { neuronsCount = 3; }
        this.neuronNet = new Array();
        this.layersCount = layersCount;
        this.neuronsCount = neuronsCount;
        this.inputs = inputs;
        for (var i = 0; i < layersCount; i++) {
            for (var j = 0; j < neuronsCount; j++) {
                this.neuronNet[i][j] = new Neuron(inputs);
            }
        }
    }
    return Perceptron;
}());
console.log('hello');
