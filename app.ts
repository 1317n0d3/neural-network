class Neuron {
  alpha: number = 1
  inputs: Array<number>
  weights: Array<number>
  output: number

  constructor(inputs: Array<number>) {
    this.inputs = inputs
    this.resetNetwork(this.weights)
    this.output = this.activate()
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

  getOutput(): number {
    return this.output
  }
}

class Perceptron {
  outputs: Array<number>
  inputs: Array<number>
  neuronNet: Array<Array<Neuron>> = new Array<Array<Neuron>>()
  // количество слоев
  layersCount: number
  // количество нейронов на каждом слое
  neuronsCount: number

  constructor(layersCount: number = 3, neuronsCount: number = 3, inputs: Array<number>) {
    this.layersCount = layersCount
    this.neuronsCount = neuronsCount
    this.inputs = inputs
    
    for(let i = 0; i < layersCount; i++) {
      for(let j = 0; j < neuronsCount; j++) {
        this.neuronNet[i][j] = new Neuron(inputs);
      }
    }
  }
}

console.log('hello');
