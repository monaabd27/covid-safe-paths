import '@tensorflow/tfjs-react-native';
import 'react-native-fs';

import * as tf from '@tensorflow/tfjs';
import { asyncStorageIO } from '@tensorflow/tfjs-react-native';
import React from 'react';

export default function Model(filename) {
  let data = require(filename);
  //let data = require('./sample_exposure_data.json')
  let dat = JSON.stringify(data);
  let da = JSON.parse(dat);

  const inps = [];
  const outs = [];
  for (let i in da) {
    let temp = [];
    for (let j in da[i]) {
      temp.push(parseFloat(JSON.stringify(da[i][j])));

      //inps.push(JSON.stringify(da[i]));
    }
    outs.push(temp[3]);
    temp.splice(3, 1);
    inps.push(temp);
  }

  //   async function waitForTensorFlowJs() {
  //     await tf.ready();
  //     setTfReady(true);
  //   }
  // waitForTensorFlowJs();

  tf.setBackend('cpu');
  tf.ready();

  const fs = require('fs');

  const path = './trained_model';
  const xs = tf.tensor2d(inps);
  const ys = tf.tensor2d(outs, [da.length, 1]);

  async function save_and_update() {
    try {
      if (fs.existsSync(path)) {
        //if file exists, average the two models
        const old = await tf.loadLayersModel(asyncStorageIO('trained_model'));
        old.fit(xs, ys, { epochs: 10 });
      }
    } catch (err) {
      //if it doesn't exst, save the model locally
      const client = tf.sequential();
      client.add(tf.layers.dense({ units: 10, inputShape: [3] }));
      client.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

      console.log(inps[1]);
      client.fit(xs, ys, { epochs: 10 });
      await client.model.save(asyncStorageIO('trained_model'));
    }

    //For future: get the trained model from server
    //and train on it locally, then send back to server
  }

  save_and_update();

  return <></>;

  // const styles = StyleSheet.create({
  //   container: {
  //     flex: 1,
  //     justifyContent: 'center',
  //     alignItems: 'center',
  //     backgroundColor: '#F5FCFF',
  //   },
  //   welcome: {
  //     fontSize: 20,
  //     textAlign: 'center',
  //     margin: 10,
  //   },
  //   instructions: {
  //     textAlign: 'center',
  //     color: '#333333',
  //     marginBottom: 5,
  //   },
  // });
  // }
}
