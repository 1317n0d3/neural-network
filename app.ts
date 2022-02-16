class Perceptron {
  alpha: number = 1
  inputs: Array<number>
  weights: Array<number>
  outputs: Array<number>
  neurons: Array<number>
  layersCount: number
  neuronsCount: Array<number>

  constructor(inputs: Array<number>, weights: Array<number>, outputs: Array<number>, 
    layersCount: number, ...neuronsCount: Array<number>) {
    this.inputs = inputs
    this.weights = weights
    this.outputs = outputs
    this.layersCount = layersCount
    this.neuronsCount = neuronsCount
  }

  getWeightSum(inputs: Array<number>): number {
    return this.weights.reduce((a, b, index) => a + b * inputs[index])
  }

  activate(): number {
    return 1 / (1 + Math.exp(-this.alpha * this.getWeightSum(this.inputs)))
  }
}

const neuron = new Perceptron([1, 2, 3, 4], [1, 2, 3, 4], [3], 1)

console.log(neuron.activate())