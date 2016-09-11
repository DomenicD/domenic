export type NetworkCommand = 'forward_pass' | 'backward_pass' | 'adjust_weights' |
                      'adjust_biases' | 'adjust_parameters';

export type TrainerCommand = 'batch_train' | 'single_train' | 'validate';

export enum NetworkType {
  STANDARD_FEED_FORWARD,
  QUADRATIC_FEED_FORWARD
}

export enum TrainerType {
  CLOSED_FORM_FUNCTION
}

export interface ParameterSet {
  name: string;
  dimensionDepth: number;
  values: number[] | number[][];
  gradients: number[] | number[][];
  deltas: number[] | number[][];
}

export interface ParameterSetMap {
  [key: string]: ParameterSet;
}

export interface NeuralNetwork {
  id: string;
  totalError: number;
  inputCount: number;
  outputCount: number;
  layerCount: number;
  parameters: ParameterSetMap[];
}

export interface TrainerBatchResult {
  batchNumber: number;
  batchSize: number;
  totalError: number;
  avgError: number;
  parameters: ParameterSetMap[];
  inputs: number[][];
  expected: number[][];
  actual: number[][];
}

export interface TrainerValidationResult {
  inputs: number[][];
  expected: number[][];
  actual: number[][];
  error: number;
}

export interface Trainer {
  id: string;
  networkId: string;
  batchSize: number;
  stepTally: number;
  batchTally: number;
}

