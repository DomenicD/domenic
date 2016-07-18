interface FeedForward {
  id: string;
  biases: number[][];
  weights: number[][][];
  inputs: number[][];
  outputs: number[][];
  node_errors: number[][];
  weight_gradients: number[][];
  bias_gradients: number[][];
  total_error: number;
}

function toNumbers(list: Array<number | string>): number[] {
  return list.map((str: number | string) => Number(str));
}

function toNumber(value: string | number): number { return Number(value); }

class FeedForwardDomain implements FeedForward {
  id: string;
  biases: number[][];
  weights: number[][][];
  inputs: number[][];
  outputs: number[][];
  node_errors: number[][];
  weight_gradients: number[][];
  bias_gradients: number[][];
  total_error: number;

  constructor(private neuralNetworkApi: NeuralNetworkApi, ff: FeedForward) {
    this.updateState(ff);
  }

  forwardPass(inputs: string[] | number[]): ng.IPromise<void> {
    return this.neuralNetworkApi
        .updateFeedForward(this.id, FeedForwardCommandEnum.FORWARD_PASS,
                           toNumbers(inputs))
        .then(ff => this.updateState(ff));
  }

  backwardPass(expected: number[]): ng.IPromise<void> {
    return this.neuralNetworkApi
        .updateFeedForward(this.id, FeedForwardCommandEnum.BACKWARD_PASS,
                           toNumbers(expected))
        .then(ff => this.updateState(ff));
  }

  adjustWeights(learningRate: number): ng.IPromise<void> {
    return this.neuralNetworkApi
        .updateFeedForward(this.id, FeedForwardCommandEnum.ADJUST_WEIGHTS,
                           toNumber(learningRate))
        .then(ff => this.updateState(ff));
  }

  adjustBiases(learningRate: number): ng.IPromise<void> {
    return this.neuralNetworkApi
        .updateFeedForward(this.id, FeedForwardCommandEnum.ADJUST_BIASES,
                           toNumber(learningRate))
        .then(ff => this.updateState(ff));
  }

  adjustParameters(learningRate: number): ng.IPromise<void> {
    return this.neuralNetworkApi
        .updateFeedForward(this.id, FeedForwardCommandEnum.ADJUST_PARAMETERS,
                           toNumber(learningRate))
        .then(ff => this.updateState(ff));
  }

  private updateState(ff: FeedForward) {
    this.id = ff.id;
    this.biases = ff.biases;
    this.weights = ff.weights;
    this.inputs = ff.inputs;
    this.outputs = ff.outputs;
    this.node_errors = ff.node_errors;
    this.weight_gradients = ff.weight_gradients;
    this.bias_gradients = ff.bias_gradients;
    this.total_error = ff.total_error;
  }
}

type FeedForwardCommand = 'forward_pass' | 'backward_pass' | 'adjust_weights' |
                          'adjust_biases' | 'adjust_parameters';

const FeedForwardCommandEnum = {
  get FORWARD_PASS() : FeedForwardCommand{return 'forward_pass'},
  get BACKWARD_PASS() : FeedForwardCommand{return 'backward_pass'},
  get ADJUST_WEIGHTS() : FeedForwardCommand{return 'adjust_weights'},
  get ADJUST_BIASES() : FeedForwardCommand{return 'adjust_biases'},
  get ADJUST_PARAMETERS() : FeedForwardCommand{return 'adjust_parameters'}
};

class NeuralNetworkApi {

  constructor(private $http: ng.IHttpService) {}

  createFeedForward(layers: string[] |
                    number[]): ng.IPromise<FeedForwardDomain> {
    return this
        .postRequestProcessing(this.$http.post("/create_feedforward",
                                               {layers : toNumbers(layers)}))
        .then(ff => new FeedForwardDomain(this, ff));
  }

  getFeedForward(id: string): ng.IPromise<FeedForwardDomain> {
    return this.postRequestProcessing(this.$http.get(`/get_feedforward/${id}`))
        .then(ff => new FeedForwardDomain(this, ff));
  }

  updateFeedForward(id: string, command: FeedForwardCommand,
                    ...args: any[]): ng.IPromise<FeedForward> {
    return this.postRequestProcessing(
        this.$http.post(`/update_feedforward/${id}/${command}`, {args}));
  }

  private postRequestProcessing(response: ng.IPromise<any>) {
    return response.then(r => r.data);
  }
}

class InsightController {
  private feedForward: FeedForwardDomain;

  constructor(private neuralNetworkApi: NeuralNetworkApi) {}

  createFeedForward(layers: string[]) {
    this.neuralNetworkApi.createFeedForward(layers).then(
        ff => this.feedForward = ff);
  }
}

let insight = angular.module('insight', [ 'ngMaterial' ])
                  .service('neuralNetworkApi', NeuralNetworkApi)
                  .controller('insightController', InsightController);

angular.bootstrap(document, [ insight.name ]);
