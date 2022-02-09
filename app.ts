class Neuron {
  alpha: number = 1
  beta: number = 2
  gamma: number = 3
  inputs: Array<number>
  weights: Array<number>
  output: number

  constructor(inputs: Array<number>, weights: Array<number>, output: number) {
    this.inputs = inputs;
    this.weights = weights;
    this.output = output;
  }

  getWeightSum(): number {
    return this.weights.reduce((a, b, index) => a + b * this.inputs[index]);
  }

  activate(): number {
    return 1 / (1 + Math.exp(-this.alpha * (this.getWeightSum() + this.gamma))) + this.beta;
  }
}

const neuron: Neuron = new Neuron([1, 2, 3, 4], [1, 2, 3, 4], 3);

console.log(neuron.getWeightSum());
console.log(neuron.activate());
