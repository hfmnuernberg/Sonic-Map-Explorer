const max = require('max-api');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { Midi } = require('@tonejs/midi'); // https://github.com/Tonejs/Midi
const { buildGruEncoder, buildGruDecoder, buildGRUSupervised } = require('./modulesGRU');
const { buildEncoder, buildDecoder, buildMLPSupervised, buildMLPSupervisedBig } = require('./modulesMLP');
const { generateColorPalette, hslToRgb, normalizeData, findBounds, hslHueToRgb, normalizeDataList } = require('./utilities');
// const { set } = require('express/lib/application');
let state = 1;
const stateNames = ['off', 'idle', 'recording', 'training', 'inference'];
const latentDim = 2;

let encoder = null;
let decoder = null;
let supervisedModel = null;
let inputStreamTensor = null;
// let inputStreamLength = 20;
let umapTensor = null;
let bounds = null;

let trainingData = null;
// let normalizedTrainingData = null;
// trainingData.dispose();
let UseSymmetricNormalization = true;

let batchSize = 100;
let epochs = 100;
let initialLearningRate = 0.0001;
let learningRate = initialLearningRate;

let isTraining = false;

let stopTraining = false;

let recordedData = [];
let umapData = [];
let isRecording = false;
// const channels = 1;
let minValues = [];
let maxValues = [];
let normalizedData = [];
let shouldNormalize = false; // Flag to indicate if normalization should be applied

let numFeatures = 50;
let sequenceLength = 1;

let colorData = [];
const maxColorIndex = 5;
let colorPalette = generateColorPalette(maxColorIndex);
let colorIndex = 0;

//let streamLen = 40; // Maximum number of vectors in the stream
let stream = Array(sequenceLength).fill(0).map(() => Array(numFeatures).fill(0)); // List to store the incoming vectors
// let medianStream = Array(numFeatures).fill(0); // List to store the median vectors
let isStreamUpdating = false; // Flag to indicate if the stream is being updated
//let streamCopy = Array(sequenceLength).fill(0).map(() => Array(numFeatures).fill(0));
let streamTensor = tf.zeros([1, sequenceLength, numFeatures]);
let normalizationFactors = tf.zeros([numFeatures, 2]);

let typeIndex = 3;
const typeList = ['gruVAE', 'mlpVAE', 'gruSupervised', 'mlpSupervised'];
let type = typeList[typeIndex]; // 'gru' or 'mlp'
function buildVAE(){
  if (type === 'gruVAE'){
    encoder = buildGruEncoder(latentDim, sequenceLength, numFeatures);
    decoder = buildGruDecoder(latentDim, sequenceLength, numFeatures);
    inputStreamTensor = tf.zeros([1, sequenceLength, numFeatures]);
  }
  if (type === 'mlpVAE'){
    encoder = buildEncoder(latentDim, numFeatures);
    decoder = buildDecoder(latentDim, numFeatures);
    sequenceLength = 1;
    inputStreamTensor = tf.zeros([1, numFeatures, 1]);
  }
  if (type === 'gruSupervised'){
    supervisedModel = buildGRUSupervised(latentDim, sequenceLength, numFeatures);
    max.post('Supervised GRU model built.');  
    sequenceLength = 10;
  }
  if (type === 'mlpSupervised'){
    if (numFeatures === 20){
      supervisedModel = buildMLPSupervised(latentDim, sequenceLength, numFeatures);
      max.post('Supervised MLP model built.');  
      
    } else if (numFeatures === 50){
      supervisedModel = buildMLPSupervisedBig(latentDim, sequenceLength, numFeatures);
      max.post('Big Supervised MLP model built.');  
    }
  }
  if (supervisedModel){
    supervisedModel.compile({
    optimizer: tf.train.adam(learningRate),
    // loss: tf.losses.huberLoss,
    loss: tf.losses.meanSquaredError,
    metrics: ['accuracy'] // or 'meanSquaredError'
    });
  }

}
buildVAE();
// supervised.summary();
// supervised.getWeights().forEach((weight, index) => {
//   max.post(`Weight ${index} min:`, weight.min().dataSync()[0], 'max:', weight.max().dataSync()[0]);
// });

function setParametersForGUI(){
  max.outlet('epochs', epochs);
  max.outlet('batchSize', batchSize);
  max.outlet('learningRate', learningRate);
  max.outlet('latentDim', latentDim);
  max.outlet('sequenceLength', sequenceLength);
  max.outlet('numFeatures', numFeatures);
  max.outlet('typeIndex', typeIndex);
  max.outlet('recordedDataSize', recordedData.length);
  max.outlet('umapDataSize', umapData.length);
}
setParametersForGUI();

max.addHandler('typeIndex', (newTypeIndex) => {
  typeIndex = newTypeIndex;
  type = typeList[typeIndex];
  reinitialize();
  max.post('new type:', type);
  
});

function waitForCondition(conditionFn, interval = 25) {
  return new Promise((resolve) => {
    const checkCondition = setInterval(() => {
      if (conditionFn()) {
        clearInterval(checkCondition); // Stop checking
        resolve(); // Resolve the promise
      }
    }, interval);
  });
}
// TESTING
// const data = tf.tensor([
//   [[1, 2, 3], [1, 11, 12]],
//   [[1, 5, 6], [1, 14, 15]],
//   [[1, 8, 9], [1, 17, 18]]
// ]);

// const b = findBounds(data);
// max.post('bounds: ', b.arraySync());
max.addHandler('state', (newState) => {
  const oldState = state;
  state = newState;
  max.post('new state:', stateNames[state]);
  if (oldState === 2 && state === 1){
    max.post('recordedData size:', recordedData.length);
    exportTrainingDataForUmapSync();
  }
  if (state === 3){
    //training state:
    stopTraining = false;
    trainSupervised();
  } else if (oldState === 3 && state === 1){
    stopTraining = true;
  }
});

function exportTrainingDataForUmapSync(){
  max.outlet('sequenceLength', sequenceLength);
  max.outlet('numFeatures', numFeatures);
  max.outlet('recordedDataSize', recordedData.length);
  max.outlet('umapDataSize', umapData.length);
  // max.post('Exporting training data for UMAP...');
  max.outlet('log', 'Processing ', recordedData.length, ' data points...');
  // max.post(recordedData);
  if (shouldNormalize){
  let [_normalizedData, _minValues, _maxValues] = normalizeDataList(recordedData);
  minValues = _minValues;
  maxValues = _maxValues;
  normalizedData = _normalizedData;
  } else {
  normalizedData = recordedData;
  }

  recordedData.forEach((data, i) => {
    // data is a 2d array with shape [sequenceLength, numFeatures]. 
    // in case the sequenceLength is greater than 1, mean over seqLen should be calculated.

    // max.post('Processing data for index:', i);
    // const meanRow = data.reduce((acc, val) => acc.map((x, j) => x + val[j]), Array(numFeatures).fill(0)).map(x => x / sequenceLength);
    const outputData = [i, ...data];
    // max.post('outputData: ', ...outputData);
    // max.post('data: ', data);
    max.outlet('recordedData', ...outputData);
  });
  
  max.outlet('exportDataForUmapFinished', 1);
}

max.addHandler('dataLookup', (index) => {
  if (index < 0 || index >= recordedData.length) {
    max.post('Invalid index:', index);
    return;
  }
  const data = recordedData[index];
  max.outlet('dataLookup', ...data);
}
);

max.addHandler('record', async (status) => {
    isRecording = status;
    max.post('recording status:', isRecording);

    //save data when recording is stopped
    if (!isRecording){
      await waitForCondition(() => !isStreamUpdating);
      // max.post('data size:', recordedData.length);
      // max.post('seqLen: ', recordedData[1].length);
      if (recordedData.length > 0){
            // reset trainigData and bounds:
            trainingData = null;
            bounds = null;
            // saveDataAsTensor;
            // max.post(recordedData);
            trainingData = tf.tensor(recordedData);
            // max.post(trainingData.shape);
            // max.post (trainingData.arraySync());
            bounds = findBounds(trainingData);
            trainingData = normalizeData(trainingData, bounds);
            // max.post('bounds: ', bounds.arraySync());
            // let containsNaN = trainingData.isNaN().any().dataSync()[0];
            // if (containsNaN){
            //   max.post('Error: Training data contains NaN values.');
            //   return;
            // }
            // trainingData = UseSymmetricNormalization ? normalizeDataSym(trainingData) : normalizeData(trainingData);
            
            const containsNaN = trainingData.isNaN().any().dataSync()[0];
            if (containsNaN){
              max.post('Error: Training data contains NaN values after normalization.');
              return;
            }
            
            // create color data dependent on trainingData length
            // colorData = [];
            // const trainingSize = trainingData.shape[0];



            // for (let i = 0; i < trainingData.shape[0]; i++) {
            //   const hue = (i / trainingSize) * 360;
            //   max.post('hue: ', hue);
            //   const currentColor = hslToRgb([hue, 100, 50]); // HSL to RGB conversion
            //   colorData.push([i, currentColor[0], currentColor[1], currentColor[2], 1]); // Add index and color
            // }
            


            max.post('trainingData shape: ', trainingData.shape);
            // max.post('color data length: ', colorData.length);

            exportTrainingDataForUmap();
            // max.outlet('recordingStopped', 1);
            // max.post('normalizedTrainingData shape: ', normalizedTrainingData.shape);
            // max.post('colors.length: ', colorData.length);

            
            //)
            // trainingData = type === 'gru' ? convertToTensorForGru(recordedData, sequenceLength, numFeatures) : convertToTensorForMLP(recordedData);
      }
    }
});


function streamData (data) {
  // Ensure the incoming data is a valid vector
  if (!Array.isArray(data) || data.length !== numFeatures) {
    max.post('Invalid input: Expected a vector of length ' + numFeatures);
    return;
   }
  // Add the new vector to the stream
  let scaledData = data.map(element => (element / 100));
  // clip data to [-1, 1]
  scaledData = scaledData.map(element => Math.max(-1, Math.min(1, element)));

  max.outlet('stream', ...scaledData);
  // max.post (scaledData)
  
  stream.push(scaledData);
  // Ensure the stream contains only the last `streamLen` vectors
  if (stream.length > sequenceLength) {
      stream.shift(); // Remove the oldest vector
  }
  // max.post(stream[0]);
  return scaledData;
}

max.addHandler('stream', (...data) => {
  
  if (isStreamUpdating) {
    max.post('Stream update in progress. Skipping this update.');
    return;
  }

  try {
    isStreamUpdating = true; // Lock the stream update
    

    switch (stateNames[state]) {
      case 'off':
        
        break;
      case 'idle':
        streamData(data);
        break;
      case 'recording': {
        const currentDataPoint = streamData(data);
        recordedData.push(currentDataPoint);
        break;}
      case 'training':
        break;
      case 'inference': {
        const currentDataPoint = streamData(data);
        predictSupervised(currentDataPoint);}
    }

    isStreamUpdating = false; // Release the flag
    
  } catch (error) {
    max.post('Error in stream handler:', error.message);
  } finally {
    
  }

  
});


max.addHandler('seqLen', (newSeqLen) => {
    sequenceLength = newSeqLen;
    reinitialize();
    max.post('new sequence length:', sequenceLength);
});
max.addHandler('numFeatures', (newNumFeatures) => {
    numFeatures = newNumFeatures;
    max.post('new number of features:', numFeatures);
    reinitialize();
});
max.addHandler('clearData', () => {
    recordedData = [];
    max.post('data size:', recordedData.length);
    max.post('Data cleared.');
});

max.addHandler('epochs', (new_epochs) => {
    epochs = new_epochs;
    max.post('new epochs:', epochs);
});

max.addHandler('stopTraining', () => {
  max.post ('istraining: ', isTraining);
  if (isTraining){
    stopTraining = true;
  } else {
    stopTraining = false;
  }
  
})

max.addHandler('batchSize', (batchSizeNew) => {
  if (!isTraining){

    batchSize = batchSizeNew;
    if (batchSize <= 0 || batchSize > trainingData.shape[0]){batchSize = trainingData.shape[0];}
    max.post('new batch size:', batchSize);
  } else {
    max.post('training in progress...'); 
  }

})



max.addHandler('learningRate', (newLearningRate) => {
    if (!isTraining){
        learningRate = newLearningRate;
        max.post('new learning rate:', learningRate);

        optimizer.dispose(); // Dispose of the old optimizer to free memory
        optimizer = tf.train.adam(learningRate);

    } else {
        max.post('training in progress...'); 
    }
}
);

// Optional: Add a handler to retrieve or save the recorded data
max.addHandler('saveData', (filePath) => {
    if (recordedData.length === 0) {
        max.post('No data to save.');
        return;
    }
    if (recordedData.length !== colorData.length) {
      max.post('Error: recordedData and colorData lengths do not match.');
      return;
    }

    try {
      const combinedData = {
        recordedData: recordedData,
        colorData: colorData
      };
        const jsonData = JSON.stringify(combinedData, null, 2);
        require('fs').writeFileSync(filePath, jsonData);
        max.post('Data saved to:', filePath);
    } catch (error) {
        max.post('Error saving data:', error.message);
    }
});

max.addHandler('loadData', (filePath) => {
    try {
        if (!fs.existsSync(filePath)) {
            max.post('Error: File does not exist:', filePath);
            return;
        }
        recordedData = [];
        colorData = [];
        const jsonData = fs.readFileSync(filePath, 'utf8');
        const combinedData = JSON.parse(jsonData); // Parse the JSON data into the array
        recordedData = combinedData.recordedData || [];
        colorData = combinedData.colorData || [];

        max.post('Data loaded from:', filePath);
        max.post('Loaded data size:', recordedData.length);
        max.post('Color Data size:', colorData.length);
        // convertDataToTensor(recordedData);
        // if (type === 'gru'){
        //   trainingData = convertToTensorForGru(recordedData, sequenceLength, numFeatures);
        // }else if (type === 'mlp'){
        //   trainingData = convertToTensorForMLP(recordedData);
        // }
        trainingData = tf.tensor(recordedData);
        
        // trainingData = UseSymmetricNormalization ? normalizeDataSym(trainingData) : normalizeData(trainingData);
        bounds = findBounds(trainingData);
        trainingData = normalizeData(trainingData, bounds);

        const containsNaN = trainingData.isNaN().any().dataSync()[0];
        if (containsNaN){
          max.post('Error: Training data contains NaN values.');
          return;
        }
        max.post('trainingData shape: ', trainingData.shape);
        // max.post('normalizedTrainingData shape: ', normalizedTrainingData.shape);
        exportTrainingDataForUmap();

    } catch (error) {
        max.post('Error loading data:', error.message);
    }
});

max.addHandler('saveModel', async (pathFromMax) => {
    const filepath = "file://" + pathFromMax;
    try {
      const metadata = {
        description: "Model Type: " + type,
        version: "1.0",
        createdBy: "Alexander Lunt",
        dateCreated: new Date().toISOString(),
        customInfo: {
          latentDim: latentDim,
          sequenceLength: sequenceLength,
          numFeatures: numFeatures,
          type: type
        }
      };
      if (type === 'gruVAE' || type === 'mlpVAE'){
        await encoder.save(filepath);
        max.post(`encoder model saved at: ${filepath}`);
        
      } else if (type === 'gruSupervised' || type === 'mlpSupervised'){
        await supervisedModel.save(filepath);
        max.post(`supervised model saved at: ${filepath}`);
      }

      const metadataPath = pathFromMax + '/metadata.json';
      fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
      max.post('Metadata saved to:', metadataPath);

      const umapDataPath = pathFromMax + '/umapData.json';
      max.outlet('dirnameUmap', 'save', umapDataPath);
      const colorDataPath = pathFromMax + '/colorData.json';
      max.outlet('dirnameColor', 'save', colorDataPath);
      
        
      
    } catch (error) {
        max.post('Error saving model:', error.message);
    }
});



max.addHandler('loadModel', async (pathFromMax) => {
    const filepath = "file://" + pathFromMax;
    max.post('Loading model from:', filepath);
    try {
      
      const modelFolderPath = path.dirname(pathFromMax); // Remove file:// and get the folder
      const metadataPath = path.join(modelFolderPath, 'metadata.json');
      if (fs.existsSync(metadataPath)) {
        const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
        // max.post('Metadata loaded:', JSON.stringify(metadata, null, 2));
        const customInfo = metadata.customInfo;
        if (customInfo){
          max.post('Loaded metadata:');
          max.post('latentDim:', customInfo.latentDim);
          max.post('sequenceLength:', customInfo.sequenceLength);
          max.post('numFeatures:', customInfo.numFeatures);
          max.post('type:', customInfo.type);

          // set parameters according to metadata:
          numFeatures = customInfo.numFeatures;
          max.outlet('numFeatures', numFeatures);


          // bounds = tf.tensor(customInfo.normalizationFactors);
          // if (type != customInfo.type){
          //   max.post('Error: Current Model type does not match the loaded model type.');
          //   max.post('Current model type:', type, 'Loaded model type:', customInfo.type);
          // }
        } else {
          max.post('Error: No metadata found!');
        }
        
      } else {
        max.post('No metadata found.');
      }
      
      if (type === 'gruVAE' || type === 'mlpVAE'){
        encoder = await tf.loadLayersModel(filepath);
        // retrieveMetadata(encoder);
        
      } else if (type === 'gruSupervised' || type === 'mlpSupervised'){
        max.post('Loading supervised model...');
        supervisedModel = await tf.loadLayersModel(filepath);
        if (supervisedModel){max.post('supervised model loaded.');}
        // retrieveMetadata(supervisedModel);
      }

      modelLoaded = true;  
      // max.post('Original encoder model reconstructed with sampleLayer.');
      max.outlet('modelLoaded', 1);

      const umapDataPath = path.join(modelFolderPath, 'umapData.json');
      max.outlet('dirnameUmap', 'read', umapDataPath);
      const colorDataPath = path.join(modelFolderPath, 'colorData.json');
      max.outlet('dirnameColor', 'read', colorDataPath);
      
      // dummyEncoder.dispose();
    } catch (error) {
        max.post('Error loading model:', error.message); // Print error message
        encoder = null; // Reset decoder to null if loading failed (important!)
        encoder = buildEncoder(latentDim);
        modelLoaded = false;
        max.outlet('modelLoaded', 0);
    }
});


function retrieveMetadata(model){
  const metadata = model.userDefinedMetadata;
  if (metadata) {
    max.post('metadata exists');
    max.post('Custom Metadata:', JSON.stringify(metadata, null, 2));
    const customInfo = metadata.customInfo;
    if (customInfo){
      bounds = tf.tensor(customInfo.normalizationFactors);
      if (tpye != customInfo.type){
        max.post('Error: Current Model type does not match the loaded model type.');
        max.poset('Current model type:', type, 'Loaded model type:', customInfo.type);
      }
    } else {
      max.post('No custom metadata found in the model.');
    }
}
}


max.addHandler('clear', async () => {
  // modelLoaded = false;
  // if (trainingData){
  //   trainingData.dispose();
  // }
  // numFeatures = 20;
  // recordedData = [];
  // colorData = [];
  // model = null;
  // supervisedModel = null;
  // max.post('data size:', recordedData.length);
  // max.post('Model cleared! Training data cleared!');
  // buildVAE();
  reinitialize();
});


max.addHandler('exportDataForUmap', async () => {
  if (!trainingData){
    max.post('no data loaded...');
    return;
  }

  
  exportTrainingDataForUmap();


});


async function exportTrainingDataForUmap(){
  
  const meanTensor = trainingData.mean(1); // mean over the sequence length
  max.post('mean tensor shape: ', meanTensor.shape);
  const meanData = await meanTensor.arraySync();
  meanTensor.dispose();

  const colorDataDeepCopy = JSON.parse(JSON.stringify(colorData));

  // const combinedSequencesTensor = trainingData.reshape([-1, sequenceLength*numFeatures]); // combine the sequences
  
  
  // max.post('combined sequencestensor shape: ', combinedSequencesTensor.shape);
  
  max.outlet('sequenceLength', sequenceLength);
  max.outlet('numFeatures', numFeatures);

  // const combinedSequencesData = combinedSequencesTensor.arraySync();
  // Process each element asynchronously
  await Promise.all(
    
    meanData.map(async (meanRow, i) => {
      try {
        // max.post(`Processing row ${i}`);
        meanRow.unshift(i);
        max.outlet('meanData', ...meanRow);
  
        // combinedSequencesData[i].unshift(i);  
        // max.outlet('normalizedData', ...combinedSequencesData[i]);
  
        colorDataDeepCopy[i].unshift(i);
        max.outlet('colorData', ...colorDataDeepCopy[i]);
        // max.post('colorData:', colorDataDeepCopy[i]);
        // max.post(`Finished processing row ${i}`);
      } catch (error) {
        max.post(`Error processing row ${i}: ${error.message}`);
      }
    })
  );
  max.outlet('exportDataForUmapFinished', 1);
}



// let trai = tf.tensor([
//   [[[1, 2, 3]], [[4, 5, 6]]],
//   [[[7, 8, 9]], [[10, 11, 12]]]
// ]);
// trai = trai.reshape([2, 6]);
// max.post(trai.arraySync());


max.addHandler('sampleIndex', (index) => {
  colorIndex = index -1;
  max.post('color index:', colorIndex);
  // colors contains: index, r, g, b, a


});

max.addHandler('loadUmapData', (dict) =>{
  try {
    const data = dict.data;
    // max.post('umap data:', data);

    // Check if data is an object
    if (typeof data !== 'object' || data === null) {
      max.post('Error: Data is not a valid object.');
      return;
    }
    // Convert the object into a JavaScript array
    const jsList = Object.values(data);
    // Validate the array
    if (!Array.isArray(jsList) || jsList.some(entry => !Array.isArray(entry) || entry.length !== 2)) {
      max.post('Error: Invalid data format. Each entry must be an array with 2 elements.');
      return;
    }
    umapData = jsList; // Store the loaded data in umapData
    max.post('UMAP data:', umapData.length, 'entries.');
    max.outlet('log', 'Exporting umap data finished.');
    max.outlet('log', 'umapData length:', umapData.length, 'recordedData length:', recordedData.length);

    

    // Convert the JavaScript list into a TensorFlow.js tensor
    // umapTensor = tf.tensor(jsList);
    // umapTensor = umapTensor.mul(2).sub(1);  //normalize to [-1,1]
    
    // max.post('umap tensor shape:', umapTensor.shape);
    // max.post('umap tensor:', umapTensor.arraySync());

    // post the first 10 elements of the loaded data
    // const first10Elements = jsList.slice(0, 10);
    // max.post('First 10 elements of the loaded data:', first10Elements);
  } catch (error) {
    max.post('Error processing UMAP data:', error.message);
    max.outlet('log', 'Error processing UMAP data:', error.message);
  }
});



// Call the trainModel function
max.addHandler('train', async ()=>{
  if (!isTrainingPossible()) {max.post("training not possible...");return};

  isTraining = true;   
  try{
    max.post('Training...');
    if (type === 'gruSupervised' || type === 'mlpSupervised'){
      await trainSupervised();
    } else if (type === 'gruVAE' || type === 'mlpVAE'){
      await trainVAE();
    }
  }
  catch (error) {
    max.post('Error during training:', error.message);
  }
  // finally { 
  //   isTraining = false;
  //   max.post('Training finished. isTraining set to false in finally block.');
  // }
});

function trainSupervised() {
  max.outlet('epochs', epochs);
  max.outlet('log', 'Training...');
  
  trainingData = tf.tensor(recordedData).reshape([recordedData.length, sequenceLength, numFeatures]);
  umapTensor = tf.tensor(umapData).reshape([umapData.length, latentDim]);
  // max.outlet('log', 'data shape: ', trainingData.shape);
  max.post('Training data shape:', trainingData.shape);
  max.post('UMAP data shape:', umapTensor.shape);
  if (!supervisedModel || supervisedModel.isDisposed) {
    max.outlet('log', 'Error: Supervised model not built.');
    return;
  }
  if (trainingData.shape[0] !== umapTensor.shape[0]){
    max.outlet('log', 'Error: ', Number(trainingData.shape[0]), 'umap: ', Number(umapTensor.shape[0]));
    return;
  }
  if (trainingData.shape[1] !== sequenceLength || trainingData.shape[2] !== numFeatures){
    max.outlet('log', 'Error: Training data shape does not match the expected shape.');
    return;
  }
  if (umapTensor.shape[1] !== latentDim){
    max.outlet('log', 'Error: UMAP data shape does not match the expected shape.');
    return;
  }
  supervisedModel.fit(trainingData, umapTensor, {
    epochs: epochs,
    batchSize: batchSize,
    verbose: 0,
    callbacks: {
      onBatchEnd: (batch,logs) => {
        // max.post(`Batch ${batch + 1}: loss = ${logs.loss}`);
        max.outlet('loss', logs.loss);
        max.outlet('log', logs.loss);
        if(stopTraining){
          stopTraining = false;
          isTraining = false;
          supervisedModel.stopTraining = true;
          max.post('training stopped.');

        }
      },
      onEpochEnd: (epoch, logs) => {
        // max.post(`Epoch ${epoch + 1}: loss = ${logs.loss}`);
        max.outlet('step', epoch+1);
        if(stopTraining){
          stopTraining = false;
          isTraining = false;
          supervisedModel.stopTraining = true;
          max.post('training stopped.');

        }
      }
    }
  }).then(() => {
    max.post('Supervised training completed.');
    isTraining = false;
    stopTraining = false;
    max.outlet('isTraining', 0);
    max.outlet('trainingFinished', 1);
    max.outlet('log', 'Training completed successfully.');

    trainingData.dispose();
    umapTensor.dispose();

  }).catch(error => {
    max.post('Error during supervised training:', error.message);
    isTraining = false;
    max.outlet('isTraining', 0);
  });
};

async function trainVAE() {
  let loss = null;
  let counter = 0;

  const numBatches = Math.ceil(trainingData.shape[0] / batchSize);

  const processEpoch = async (epoch) => {
    if (stopTraining) {
      max.post('training stopped.');
      stopTraining = false;
      modelLoaded = true;
      isTraining = false;
      return;
    }

    max.post('Starting epoch:', epoch);

    const processBatch = async (batch) => {
      if (stopTraining) {
        max.post('training stopped.');
        stopTraining = false;
        modelLoaded = true;
        isTraining = false;
        return;
      }

      max.post('epoch:', epoch, 'batch:', batch);
      const start = batch * batchSize;
      const end = Math.min((batch + 1) * batchSize, trainingData.shape[0]); // Handle the last batch
      // max.post('calculating start and end.')
      let batchData = null;
      if (type === 'mlpVAE'){
        batchData = trainingData.slice([start, 0, 0], [end - start, numFeatures, 1]);
        batchData = batchData.reshape([end - start, numFeatures, 1]);

      }else if (type === 'gruVAE'){
        batchData = trainingData.slice([start, 0, 0], [end - start, sequenceLength, numFeatures]);
        batchData = batchData.reshape([end - start, sequenceLength, numFeatures]);
      }
      // max.post('batch data shape  :', batchData.shape);
      // #################### TRAIN STEP: #######################################
      await trainStep(batchData, epoch);
      // ########################################################################

      batchData.dispose();

      if (batch + 1 < numBatches) {
        setImmediate(() => processBatch(batch + 1)); // Process the next batch asynchronously
      } else {
        max.post('Epoch', epoch, 'completed.');
        await max.outlet('step', epoch);

        if (epoch < epochs) {
          setImmediate(() => processEpoch(epoch + 1)); // Start the next epoch asynchronously
        } else {
          modelLoaded = true;
          isTraining = false;
          max.post('Model trained!');
        }
      }
    };

    processBatch(0); // Start processing the first batch
  };

  processEpoch(1); // Start processing the first epoch
}

function isTrainingPossible(){
  // Dependencies for all models:
  if (!trainingData || trainingData.isDisposed) {
    max.post('No input data loaded for supervised training.');
    return false;
  } 
  if (isTraining) {
    max.post('Training in progress. Please wait.');
    return false;
  }
  // check min and max values of trainingData
  const minValues = trainingData.min([0,1]);
  const maxValues = trainingData.max([0,1]);
  if (minValues.min().dataSync()[0] < -1 || maxValues.max().dataSync()[0] > 1){
    max.post('Error: Training data is not normalized between -1 and 1.');
    return false;
  }

  // dependencies unsupervised VAE Training:


  // dependencies for supervised training:
  if (type === 'gruSupervised' || type === 'mlpSupervised'){
    // check if label data is loaded:
    if (!umapTensor) {
      max.post('No UMAP data loaded for supervised training.');
      return false;
    }
    // check if model is loaded
    if (!supervisedModel){
      max.post('Error: Supervised model not built.');
      return false;
    }
    // check min and max values of umapTensor
    const minValuesUmap = umapTensor.min([0,1]);
    const maxValuesUmap = umapTensor.max([0,1]);
    if (minValuesUmap.min().dataSync()[0] < -1 || maxValuesUmap.max().dataSync()[0] > 1){
      max.post('Error: UMAP data is not normalized between -1 and 1.');
      return false;
    }
  }

  return true;
}

function reinitialize(){
  
  stream = Array(sequenceLength).fill(0).map(() => Array(numFeatures).fill(0));; // List to store the incoming vectors
  // medianStream = Array(numFeatures).fill(0);
  // streamTensor.dispose();
  recordedData = [];
  umapData = [];
  colorData = [];
  setParametersForGUI();
  buildVAE();
}









// #####################
// Training: 
// #####################


let optimizer = tf.train.adam(learningRate);

function klLossFunc(z_mean, z_log_var) {
    return tf.tidy(() => {
      let kl_loss;
      kl_loss = tf.scalar(1).add(z_log_var).sub(z_mean.square()).sub(z_log_var.exp());
      kl_loss = tf.sum(kl_loss, -1);
      kl_loss = kl_loss.mul(tf.scalar(-0.5));
      return kl_loss;
    });
  }

async function trainStep(data, epoch) {  // 'data' should be pre-normalized between 0 and 1
  const lossValue = await optimizer.minimize(() => {
    // tf.tidy is now *inside* the minimize callback
    return tf.tidy(() => { 
      
      const [z_mean, z_log_var] = encoder.apply(data);    // Use encoder.apply
      const epsilon = tf.randomNormal(z_mean.shape); // standard normal distribution acts as "true" probability distribution
      const z = z_mean.add(z_log_var.mul(0.5).exp().mul(epsilon)); //reparametrization trick
      
      
      const reconstruction = decoder.apply(z); 

      // Debug shapes and ranges
      // max.post('Data shape:', data.shape);
      max.outlet('dataShape', data.shape[0], data.shape[1], data.shape[2]);
      // max.post('Reconstruction shape:', reconstruction.shape);
      // max.post('Data min:', data.min().dataSync()[0], 'max:', data.max().dataSync()[0]);
      // max.post('Reconstruction min:', reconstruction.min().dataSync()[0], 'max:', reconstruction.max().dataSync()[0]);

      
      
      // const reconstructionLoss = tf.metrics.binaryCrossentropy(data, reconstruction).sum(); // Sum over batch for total loss
      const reconstructionLoss = tf.metrics.meanAbsoluteError(data, reconstruction).sum().div(2);
      // max.post('Reconstruction Loss:', reconstructionLoss.dataSync()[0]);
      max.outlet('reconstructionLoss', reconstructionLoss.dataSync()[0]);
      // let reconstructionLoss = reconstructionLossFunc(data, reconstruction).sum();
      
      const klWeight = tf.scalar(epoch / epochs);
      const klLoss = klLossFunc(z_mean, z_log_var).sum().mul(klWeight); // Sum over batch for KL loss
      max.outlet('klLoss', klLoss.dataSync()[0]);
      max.outlet('z_mean', z_mean.mean().dataSync()[0]);
      max.outlet('z_log_var', z_log_var.mean().dataSync()[0]);
      // Consider KL annealing here if desired (see previous responses)
      const totalLoss = reconstructionLoss.add(klLoss).div(data.shape[0]); // Added summed losses and divided by batch size to give per-example loss.
      // max.post('Reconstruction Loss:', reconstructionLoss.dataSync()[0]);
      // max.post('KL Loss:', klLoss.dataSync()[0]);
      // max.post('Total Loss:', totalLoss.dataSync()[0]);
      // max.post('total loss:', totalLoss);
      return totalLoss;  // Return total loss to optimizer.minimize
    });
    }, true); // returnCost: true to get the loss value

    max.outlet('loss', lossValue.dataSync()[0]); // output the loss
}











// #####################
// Inference: 
// #####################
let isFree = true;
let modelLoaded = true;


function isPredictionPossible(){
  if (type === 'gruVAE' || type === 'mlpVAE'){
    if (!encoder) { // Check if the encoder is loaded
      
      max.post('Error: Model not loaded!');
      return false;
    }

    if (!modelLoaded) {
        max.post('Error (in predict handler): Model not fully loaded yet.');
        return false;
    }
  }

  if (type === 'gruSupervised' || type === 'mlpSupervised'){
    if (!supervisedModel){
      max.post('Error: Model not loaded!');
      return false;
    }

  }

  if (isTraining){max.post('training in progress...'); return false;}

  if (!isFree) {
    max.post('Blocking! Prediction in progress, please wait.');
    return false;
  }

  if (isStreamUpdating){
    max.post('Blocking! Stream update in progress, please wait.');
    return false;
  }

  // Check for Null values in the input tensor
  const inputTensorArray = streamTensor.arraySync().flat();
    const containsNull = inputTensorArray.some(element => element === null);
    if (containsNull){
      max.post('Contains null:', containsNull);
    }


  isFree = false;
  return true;
}

async function predictVAE(){
  const [z_mean, z_log_var] = encoder.predict(streamTensor);
    const epsilon = tf.randomNormal(z_mean.shape);
    const z = z_mean.add(z_log_var.mul(0.5).exp().mul(epsilon));

    // const z_sampleArray = z.arraySync().flat();
    // max.outlet("encoderOutput", z_sampleArray[0], z_sampleArray[1]);
    max.outlet("predictionOutput", z_mean.dataSync()[0], z_log_var.dataSync()[1]);

    z.dispose();
    epsilon.dispose();
    // encoderInput.dispose();
    // Clean up tensors
    z_mean.dispose();
    z_log_var.dispose();
}

function normalizeStreamVector(vector, minValues, maxValues) {
  // vector: Array of length numFeatures
  // minValues, maxValues: Arrays of length numFeatures
  return vector.map((value, i) => {
    const min = minValues[i];
    const max = maxValues[i];
    let norm = (value - min) / (max - min + 1e-8);
    norm = norm * 2 - 1; // scale to [-1, 1]
    return norm;
  });
}

// let normalizeInputTensorForPrediction = true;
function predictSupervised(modelInput){

  try{
    if (shouldNormalize){
      modelInput = normalizeStreamVector(modelInput, minValues, maxValues);
    } 
    
    streamTensor = tf.tensor(modelInput).reshape([1, sequenceLength, numFeatures]);
    let prediction = supervisedModel.predict(streamTensor);
    const predictionArray = prediction.arraySync().flat();
    // max.post('prediction:', predictionArray);
    // max.post(prediction.shape);
    max.outlet("predictionOutput", predictionArray[0], predictionArray[1]);
    prediction.dispose();

    isFree = true;
  }catch(error){
    max.post(`Prediction error catched: ${error.message}`);
  }
}


max.addHandler('predict', async () => {
  if (!isPredictionPossible()){return;}
  try{
    if (type === 'gruVAE' || type === 'mlpVAE'){
      await predictVAE();
    } else if (type === 'gruSupervised' || type === 'mlpSupervised'){
      await predictSupervised();
    }
  
    
    isFree = true;

  }catch(error){
    max.post(`Prediction error: ${error.message}`);
  }
});