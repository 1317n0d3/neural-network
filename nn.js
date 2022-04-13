const mmap = math.map;
const rand = math.random;
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

class NeuralNetwork {
  constructor(inputnodes, hiddennodes, outputnodes, learningrate, wih, who) {
    this.inputnodes = inputnodes;
    this.hiddennodes = hiddennodes;
    this.outputnodes = outputnodes;
    this.learningrate = learningrate;

    this.wih = wih || sub(mat(rand([hiddennodes, inputnodes])), 0.5);
    this.who = who || sub(mat(rand([outputnodes, hiddennodes])), 0.5);

    this.act = (matrix) => mmap(matrix, (x) => 1 / (1 + Math.exp(-x)));
  }

  static normalizeData = (data) => {
    return data.map((e) => (e / 255) * 0.99 + 0.01);
  };

  cache = { loss: [] };

  forward = (input) => {
    const wih = this.wih;
    const who = this.who;
    const act = this.act;

    input = transp(mat([input]));

    const h_in = e("wih * input", { wih, input });
    const h_out = act(h_in);

    const o_in = e("who * h_out", { who, h_out });
    const actual = act(o_in);

    this.cache.input = input;
    this.cache.h_out = h_out;
    this.cache.actual = actual;

    return actual;
  };

  backward = (target) => {
    const who = this.who;
    const input = this.cache.input;
    const h_out = this.cache.h_out;
    const actual = this.cache.actual;

    target = transp(mat([target]));

    const dEdA = sub(target, actual);

    const o_dAdZ = e("actual .* (1 - actual)", {
      actual,
    });

    const dwho = e("(dEdA .* o_dAdZ) * h_out'", {
      dEdA,
      o_dAdZ,
      h_out,
    });

    const h_err = e("who' * (dEdA .* o_dAdZ)", {
      who,
      dEdA,
      o_dAdZ,
    });

    const h_dAdZ = e("h_out .* (1 - h_out)", {
      h_out,
    });

    const dwih = e("(h_err .* h_dAdZ) * input'", {
      h_err,
      h_dAdZ,
      input,
    });

    this.cache.dwih = dwih;
    this.cache.dwho = dwho;
    this.cache.loss.push(sum(sqr(dEdA)));
  };

  update = () => {
    const wih = this.wih;
    const who = this.who;
    const dwih = this.cache.dwih;
    const dwho = this.cache.dwho;
    const r = this.learningrate;

    this.wih = e("wih + (r .* dwih)", { wih, r, dwih });
    this.who = e("who + (r .* dwho)", { who, r, dwho });
  };

  predict = (input) => {
    return this.forward(input);
  };

  train = (input, target) => {
    this.forward(input);
    this.backward(target);
    this.update();
  };
}

const inputnodes = 784;
const hiddennodes = 100;
const outputnodes = 10;
const learningrate = 0.2;
let iter = 0;
const iterations = 5;

const trainingDataPath = "./mnist/mnist_train.csv";
const testDataPath = "./mnist/mnist_test.csv";
const weightsFilename = "weights.json";
const savedWeightsPath = `./dist/${weightsFilename}`;

const trainingData = [];
const trainingLabels = [];
const testData = [];
const testLabels = [];
const savedWeights = {};

const printSteps = 1;

let myNN;

window.onload = async () => {
  myNN = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate);

  trainButton.disabled = true;
  testButton.disabled = true;
  loadWeightsButton.disabled = true;

  status.innerHTML = "Loading the data sets. Please wait ...<br>";

  const trainCSV = await loadData(trainingDataPath, "CSV");

  if (trainCSV) {
    prepareData(trainCSV, trainingData, trainingLabels);
    status.innerHTML += "Training data successfully loaded...<br>";
  }

  const testCSV = await loadData(testDataPath, "CSV");

  if (testCSV) {
    prepareData(testCSV, testData, testLabels);
    status.innerHTML += "Test data successfully loaded...<br>";
  }

  if (!trainCSV || !testCSV) {
    status.innerHTML +=
      "Error loading train/test data set. Please check your file path! If you run this project locally, it needs to be on a local server.";
    return;
  }

  trainButton.disabled = false;
  testButton.disabled = false;

  const weightsJSON = await loadData(savedWeightsPath, "JSON");

  if (weightsJSON) {
    savedWeights.wih = weightsJSON.wih;
    savedWeights.who = weightsJSON.who;
    loadWeightsButton.disabled = false;
  }

  status.innerHTML += "Ready.<br><br>";
};

async function loadData(path, type) {
  try {
    const result = await fetch(path, {
      mode: "no-cors",
    });

    switch (type) {
      case "CSV":
        return await result.text();
        break;
      case "JSON":
        return await result.json();
        break;
      default:
        return false;
    }
  } catch {
    return false;
  }
}

function prepareData(rawData, target, labels) {
  rawData = rawData.split("\n");
  rawData.pop();

  rawData.forEach((current) => {
    let sample = current.split(",").map((x) => +x);

    labels.push(sample[0]);
    sample.shift();

    sample = NeuralNetwork.normalizeData(sample);

    target.push(sample);
  });
}

function train() {
  trainButton.disabled = true;
  testButton.disabled = true;
  loadWeightsButton.disabled = true;
  download.innerHTML = "";

  if (iter < iterations) {
    iter++;
    status.innerHTML += "Starting training ...<br>";
    status.innerHTML += "Iteration " + iter + " of " + iterations + "<br>";

    trainingData.forEach((current, index) => {
      setTimeout(() => {
        const label = trainingLabels[index];
        const oneHotLabel = Array(10).fill(0);
        oneHotLabel[label] = 0.99;

        myNN.train(current, oneHotLabel);

        if (index > 0 && !((index + 1) % printSteps)) {
          status.innerHTML += `finished  ${index + 1}  samples ... <br>`;
        }

        if (index === trainingData.length - 1) {
          status.innerHTML += `Loss:  ${
            sum(myNN.cache.loss) / trainingData.length
          }<br><br>`;
          myNN.cache.loss = [];

          test("", true);
        }
      }, 0);
    });
  }
}

function test(_, inTraining = false) {
  trainButton.disabled = true;
  testButton.disabled = true;
  loadWeightsButton.disabled = true;

  status.innerHTML += "Starting testing ...<br>";

  let correctPredicts = 0;
  testData.forEach((current, index) => {
    setTimeout(() => {
      const actual = testLabels[index];

      const predict = formatPrediction(myNN.predict(current));
      predict === actual ? correctPredicts++ : null;

      if (index > 0 && !((index + 1) % printSteps)) {
        status.innerHTML += " finished " + (index + 1) + " samples ...<br>";
      }

      if (index >= testData.length - 1) {
        status.innerHTML +=
          "Accuracy: " +
          Math.round((correctPredicts / testData.length) * 100) +
          " %<br><br>";

        if (iter + 1 > iterations) {
          createDownloadLink();
          enableAllButtons();
          status.innerHTML += "Finished training.<br><br>";
          iter = 0;
        } else if (inTraining) {
          train();
        } else {
          enableAllButtons();
        }
      }
    }, 0);
  });
}

function predict() {
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 250, 250, 0, 0, 28, 28);

  const img = tempCtx.getImageData(0, 0, 28, 28);

  let sample = [];
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    sample[j] = (img.data[i + 0] + img.data[i + 1] + img.data[i + 2]) / 3;
  }

  img.data = NeuralNetwork.normalizeData(img.data);

  const predict = formatPrediction(myNN.predict(sample));
  prediction.innerHTML = predict;
}

function formatPrediction(prediction) {
  const flattened = prediction.toArray().map((x) => x[0]);

  return flattened.indexOf(Math.max(...flattened));
}

function loadWeights() {
  myNN.wih = savedWeights.wih;
  myNN.who = savedWeights.who;
  status.innerHTML += "Weights successfully loaded.";
}

function createDownloadLink() {
  const wih = myNN.wih.toArray();
  const who = myNN.who.toArray();
  const weights = { wih, who };
  download.innerHTML = `<a download="${weightsFilename}" id="downloadLink" href="data:text/json;charset=utf-8,${encodeURIComponent(
    JSON.stringify(weights)
  )}">Download model weights</a>`;
}

function enableAllButtons() {
  trainButton.disabled = false;
  testButton.disabled = false;
  loadWeightsButton.disabled = false;
}
