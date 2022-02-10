class Neuron {
  alpha: number = 1
  inputs: Array<number>
  weights: Array<number>
  outputs: Array<number>

  constructor(inputs: Array<number>, weights: Array<number>, outputs: Array<number>) {
    this.inputs = inputs
    this.weights = weights
    this.outputs = outputs
  }

  getWeightSum(inputs: Array<number>): number {
    return this.weights.reduce((a, b, index) => a + b * inputs[index])
  }

  activate(): number {
    return 1 / (1 + Math.exp(-this.alpha * this.getWeightSum(this.inputs)))
  }
}

const neuron: Neuron = new Neuron([1, 2, 3, 4], [1, 2, 3, 4], [3])

console.log(neuron.activate())