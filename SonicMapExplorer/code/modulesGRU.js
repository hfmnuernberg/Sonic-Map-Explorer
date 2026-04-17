const tf = require('@tensorflow/tfjs');
const max = require('max-api');
// Adjust these to match your encoder setup
// const sequenceLength = 20;  // <-- replace with your actual value
const num_units1 = 32;      // <-- adjust to match encoder GRU units
const num_units2 = 16;     // <-- adjust to match encoder GRU units
// const numFeatures = 47;


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

// === GRU SUPERVISED ===
function buildGRUSupervised(latentDim, sequenceLength=20, numFeatures=47) {
  const gruInputs = tf.input({ shape: [sequenceLength, numFeatures] });
  
  let x = tf.layers.gru({
    units: 40,
    returnSequences: true,
    activation: 'tanh'
  }).apply(gruInputs);

  // x = tf.layers.gru({
  //   units: 30,
  //   returnSequences: true,
  //   activation: 'tanh'
  // }).apply(x);

  x = tf.layers.gru({
    units: 20,
    returnSequences: false,
    activation: 'tanh'
  }).apply(x);

  outputLayer = tf.layers.dense({ units: latentDim, activation: 'tanh' }).apply(x);

  return tf.model({ inputs: gruInputs, outputs: outputLayer, name: 'gruSupervised' });
  // return seqModel;
}


// === GRU ENCODER ===
function buildGruEncoder(latentDim, sequenceLength=20, numFeatures=47) {
    const encoderInputs = tf.input({ shape: [sequenceLength, numFeatures] });
  
    let x = tf.layers.gru({
      units: num_units1,
      returnSequences: true,
      activation: 'tanh'
    }).apply(encoderInputs);
  
    x = tf.layers.gru({
      units: num_units2,
      returnSequences: false,
      activation: 'tanh'
    }).apply(x);
  
    const z_mean = tf.layers.dense({ units: latentDim, name: 'z_mean' }).apply(x);
    const z_log_var = tf.layers.dense({ units: latentDim, name: 'z_log_var' }).apply(x);
  
    // Assumes sampleLayer is defined externally or imported
    // const z = new sampleLayer().apply([z_mean, z_log_var]);
  
    return tf.model({ inputs: encoderInputs, outputs: [z_mean, z_log_var], name: 'encoder' });
  }


function buildDummyGruEncoder(latentDim, sequenceLength=20, numFeatures=47) {
    let encoderInputs = tf.input({ shape: [sequenceLength, numFeatures] });
    // encoderInputs = tf.layers.repeatVector({ n: sequenceLength }).apply(encoderInputs);
    let x = tf.layers.gru({
      units: num_units2,
      returnSequences: true,
      activation: 'tanh'
    }).apply(encoderInputs);
  
    x = tf.layers.gru({
      units: num_units1,
      returnSequences: false,
      activation: 'tanh'
    }).apply(x);
    // x = tf.layers.dense(20).apply(x)
    const z_mean = tf.layers.dense({ units: latentDim, name: 'z_mean' }).apply(x);
    const z_log_var = tf.layers.dense({ units: latentDim, name: 'z_log_var' }).apply(x);
  
    // Assumes sampleLayer is defined externally or imported
    // const z = new sampleLayer().apply([z_mean, z_log_var]);
  
    // return tf.model({ inputs: encoderInputs, outputs: [z_mean, z_log_var, z], name: 'encoder' });
    return tf.model({ inputs: encoderInputs, outputs: [z_mean, z_log_var], name: 'dummy_encoder' });
  }

// === GRU DECODER ===
function buildGruDecoder(latentDim, sequenceLength=20, numFeatures=47) {
  const latentInputs = tf.input({ shape: [latentDim] });

  // Expand latent vector into a sequence
  let x = tf.layers.repeatVector({ n: sequenceLength }).apply(latentInputs);

  // GRU layers to generate sequence
  x = tf.layers.gru({
    units: num_units2,
    returnSequences: true,
    activation: 'tanh'
  }).apply(x);

  x = tf.layers.gru({
    units: num_units1,
    returnSequences: true,
    activation: 'tanh'
  }).apply(x);

  // TimeDistributed Dense layer to output per time step
  const decoderOutputs = tf.layers.timeDistributed({
    layer: tf.layers.dense({ units: numFeatures, activation: 'sigmoid' }) // or 'linear' based on your data
  }).apply(x);
  // const decoderOutputs = tf.layers.dense({ units: numFeatures, activation: 'sigmoid' }).apply(x);

  return tf.model({ inputs: latentInputs, outputs: decoderOutputs, name: 'decoder' });
}


function convertToTensorForGru(data, sequenceLength = 20, numFeatures = 47) {
    const flatData = data.flat();

    // Ensure the data length is divisible by numFeatures
    if (flatData.length % numFeatures !== 0) {
        max.post('Error: Data length is not divisible by the number of features.');
        return;
    }

    // Reshape flatData into [numTimestamps, numFeatures]
    const numTimestamps = flatData.length / numFeatures;
    const reshapedData = tf.tensor(flatData, [numTimestamps, numFeatures]);

    // Calculate the number of sequences
    const numSequences = Math.max(0, numTimestamps - sequenceLength + 1);

    // Create a sliding window of shape [numSequences, sequenceLength, numFeatures]
    const sequences = [];
    for (let i = 0; i < numSequences; i++) {
        const window = reshapedData.slice([i, 0], [sequenceLength, numFeatures]);
        sequences.push(window);
    }

    // Stack the sequences into a single tensor
    const trainingData = tf.stack(sequences);

    // Dispose of intermediate tensor
    reshapedData.dispose();

    // Log the shape of the tensor
    max.post('Tensor created with shape:', trainingData.shape);

    // Return the training data tensor
    return trainingData;
}



module.exports = {
    buildGruEncoder,
    buildGruDecoder,
    convertToTensorForGru,
    buildDummyGruEncoder,
    buildGRUSupervised
  };


