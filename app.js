var Neuron = /** @class */ (function () {
    function Neuron(inputs, weights, outputs) {
        this.alpha = 1;
        this.inputs = inputs;
        this.weights = weights;
        this.outputs = outputs;
    }
    Neuron.prototype.getWeightSum = function () {
        var _this = this;
        return this.weights.reduce(function (a, b, index) { return a + b * _this.inputs[index]; });
    };
    Neuron.prototype.activate = function () {
        return 1 / (1 + Math.exp(-this.alpha * this.getWeightSum()));
    };
    return Neuron;
}());
var neuron = new Neuron([1, 2, 3, 4], [1, 2, 3, 4], [3]);
console.log(neuron.getWeightSum());
console.log(neuron.activate());
