class Neuron {
  alpha: number = 1
  inputs: Array<number>
  weights: Array<number>
  output: number

  constructor(inputs: Array<number>, output: number) {
    this.inputs = inputs
    this.output = output
    this.resetNetwork(this.weights)
  }

  getWeightSum(inputs: Array<number>): number {
    return this.weights.reduce((a, b, index) => a + b * inputs[index])
  }

  resetNetwork(weights: Array<number>): void {
    this.inputs.forEach(() => weights.push(Math.random()))
  }

  activate(): number {
    return 1 / (1 + Math.exp(-this.alpha * this.getWeightSum(this.inputs)))
  }
}

class Perceptron {
  outputs: Array<number>
  neurons: Array<number>
  layersCount: number
  neuronsCount: Array<number>

  constructor() {}
}