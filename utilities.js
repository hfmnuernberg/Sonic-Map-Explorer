const tf = require('@tensorflow/tfjs');
const max = require('max-api');

function generateColorPalette(x) {
    const colors = [];
    const saturation = 80; // High saturation for vibrant colors
    const lightness = 50;  // Medium lightness for good contrast with white
  
    for (let i = 0; i < x; i++) {
      const hue = Math.round((360 / x) * i); // Distribute hues evenly
      // colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
      colors.push([hue, saturation, lightness]);
    }
  
    return colors;
  }

function hslToRgb(hsl) {
  let [h, s, l] = hsl;

  s /= 100;
  l /= 100;
  

  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));

  // return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
  return [f(0), f(8), f(4)];
}


function hslHueToRgb(h) {
  // h in [0,1], s=1, l=0.5
  const s = 1;
  const l = 0.5;

  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;

  const r = hue2rgb(p, q, h + 1/3);
  const g = hue2rgb(p, q, h);
  const b = hue2rgb(p, q, h - 1/3);

  return [r, g, b];
}
// #########################################################
// normalization is not consistent!!
// norm factor needs to be saved with model.
// every feature has its own normalization factor.
// #########################################################


// function normalizeDataSym(data) {
//     return tf.tidy(() => {
//       const minValues = data.min([0, 1]);
//       const maxValues = data.max([0, 1]);
//       const epsilon = tf.scalar(1e-8);
//       let normalized = data.sub(minValues).div(maxValues.sub(minValues).add(epsilon));
//       normalized = normalized.mul(2).sub(1);
//       return normalized; // Return the normalized tensor
//     });
//   }

function findBounds(data) {
  return tf.tidy(() => {
    const reshaped = data.reshape([-1, data.shape[2]]);
    const minValues = reshaped.min(0);
    let maxValues = reshaped.max(0);
    let differences = maxValues.sub(minValues); // Avoid division by zero
    const epsilon = tf.scalar(1e-8);
    // Replace all values != 0 with 0, and all values == 0 with epsilon
    const bias = tf.where(differences.equal(0), epsilon, tf.zerosLike(differences));
    differences = differences.add(bias);
    maxValues = maxValues.add(bias);
    const bounds = tf.stack([minValues, maxValues, differences], axis=0); // Stack along axis 0
    return bounds; // Shape: [2, numFeatures]
  });
}

function normalizeData (data, bounds, symmetric=true){
  return tf.tidy(() => {
    const minValues = bounds.slice([0, 0], [1, -1]);
    const maxValues = bounds.slice([1, 0], [1, -1]);
    let normalized = data.sub(minValues).div(maxValues.sub(minValues));
    if (symmetric){
      normalized = normalized.mul(2).sub(1);
    }
    return normalized;
  });
}

function normalizeDataList(data, symmetric = true) {
  // data: Array of arrays (shape: [numRows, numFeatures])
  const numFeatures = data[0].length;
  // Calculate min and max for each feature
  const _minValues = Array(numFeatures).fill(Infinity);
  const _maxValues = Array(numFeatures).fill(-Infinity);

  data.forEach(row => {
    row.forEach((value, j) => {
      if (value < _minValues[j]) _minValues[j] = value;
      if (value > _maxValues[j]) _maxValues[j] = value;
    });
  });

  // Normalize data
  const _normalized = data.map(row =>
    row.map((value, j) => {
      const min = _minValues[j];
      const max = _maxValues[j];
      let norm = (value - min) / (max - min + 1e-8);
      if (symmetric) {
        norm = norm * 2 - 1;
      }
      return norm;
    })
  );

  // return { normalized, minValues, maxValues };
  return [_normalized, _minValues, _maxValues];
}

module.exports = {
    generateColorPalette,
    hslToRgb,
    normalizeData,
    // normalizeDataSym,
    findBounds,
    hslHueToRgb,
    normalizeDataList
};