/**
 * VAE with Feedforward Neural Network
 */
const tf = require('@tensorflow/tfjs');
const max = require('max-api');

// === MLP SUPERVISED ===
function buildMLPSupervised(latentDim, sequenceLength=20, numFeatures = 20) {
  const inputs = tf.input({ shape: [sequenceLength, numFeatures] });
  let x = tf.layers.flatten().apply(inputs);
  x = tf.layers.dense({ units: 100, activation: 'linear' }).apply(x);
  x = tf.layers.dense({ units: 50, activation: 'linear' }).apply(x);
  const output = tf.layers.dense({ units: latentDim, activation: 'tanh' }).apply(x);
  return tf.model({ inputs: inputs, outputs: output, name: 'mlpSupervised' });
  
}
function buildMLPSupervisedBig(latentDim, sequenceLength=20, numFeatures = 50) {
  const inputs = tf.input({ shape: [sequenceLength, numFeatures] });
  let x = tf.layers.flatten().apply(inputs);
  x = tf.layers.dense({ units: 50, activation: 'linear' }).apply(x);
  x = tf.layers.dense({ units: 25, activation: 'linear' }).apply(x);
  x = tf.layers.dense({ units: 10, activation: 'linear' }).apply(x);
  x = tf.layers.dense({ units: 6, activation: 'linear' }).apply(x);
  x = tf.layers.dense({ units: 4, activation: 'linear' }).apply(x);
  const output = tf.layers.dense({ units: latentDim, activation: 'sigmoid' }).apply(x);
  return tf.model({ inputs: inputs, outputs: output, name: 'mlpSupervisedBig' });

}


// === VAE UNSUPERVISED ===
// ENCODER
function buildEncoder(latentDim, numFeatures = 20) {
  const encoderInputs = tf.input({ shape: [numFeatures, 1] });
  let x = tf.layers.flatten().apply(encoderInputs);
  x = tf.layers.dense({ units: 100, activation: 'relu' }).apply(x);
  x = tf.layers.dense({ units: 50, activation: 'relu' }).apply(x);

  const z_mean = tf.layers
      .dense({ units: latentDim, name: 'z_mean' })
      .apply(x);
  const z_log_var = tf.layers
      .dense({ units: latentDim, name: 'z_log_var' })
      .apply(x);

  // Assumes sampleLayer is defined externally or imported
  const z = new sampleLayer().apply([z_mean, z_log_var]);

  return tf.model({ inputs: encoderInputs, outputs: [z_mean, z_log_var, z], name: 'encoder' });
}
  
  // DECODER
  function buildDecoder(latentDim, numFeatures=20) {
    const latentInputs = tf.input({ shape: [latentDim] });
    let x = tf.layers.dense({ units: 50, activation: 'relu' }).apply(latentInputs);
    x = tf.layers.dense({units: 100, activation: 'relu'}).apply(x);
    x = tf.layers.dense({units: numFeatures, activation: 'relu'}).apply(x);
    const decoderOutputs = tf.layers.reshape({ targetShape: [numFeatures, 1] }).apply(x);
    return tf.model({ inputs: latentInputs, outputs: decoderOutputs, name: 'decoder' });
  }

class sampleLayer extends tf.layers.Layer {
    constructor(args) {
      super({});
    }
  
    computeOutputShape(inputShape) {
      return inputShape[0];
    }
  
    call(inputs, kwargs) {
      return tf.tidy(() => {
        const [zMean, zLogVar] = inputs;
        const batch = zMean.shape[0];
        const dim = zMean.shape[1];
        const epsilon = tf.randomNormal([batch, dim]);
        const half = tf.scalar(0.5);
        const temp = zLogVar.mul(half).exp().mul(epsilon);
        const sample = zMean.add(temp);
        return sample;
      });
    }
  
    getClassName() {
      return 'sampleLayer';
    }
  }

function convertToTensorForMLP(data) {
  const flatData = data.flat();
  trainingData = tf.tensor(flatData, [data.length, data[0].length, 1]);
  
  max.post('Tensor created for MLP with shape:', trainingData.shape);
  return trainingData;
}
  module.exports = {
    buildMLPSupervised,
    buildEncoder,
    buildDecoder,
    convertToTensorForMLP,
    buildMLPSupervisedBig
  };