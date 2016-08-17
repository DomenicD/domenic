interface ParameterSet {
  name: string;
  dimensionDepth: number, values: number[] | number[][];
  gradients: number[] | number[][];
  deltas: number[] | number[][];
}

interface ParameterSetMap {
  [key: string]: ParameterSet;
}

interface NeuralNetwork {
  id: string;
  totalError: number;
  inputCount: number;
  outputCount: number;
  layerCount: number;
  parameters: ParameterSetMap[];
}

interface TrainerBatchResult {
  batchNumber: number;
  batchSize: number;
  totalError: number;
  avgError: number;
  parameters: ParameterSetMap[];
}

interface Trainer {
  id: string;
  networkId: string;
  batchSize: number;
  stepTally: number;
  batchTally: number;
batchResults: TrainerBatchResult[];
}

function toNumbers(list: Array<number | string>): number[] {
  return list.map((str: number | string) => toNumber(str));
}

function toNumber(value: string | number): number { return Number(value); }

abstract class DomainObject<T> {
  constructor(protected insightApi: InsightApi, protected response: T) {}

  protected postRequestProcessing(responsePromise: ng.IPromise<T>):
      ng.IPromise<void> {
    return responsePromise.then(response => {this.response = response});
  }
}

class NeuralNetworkDomain extends DomainObject<NeuralNetwork> implements
    NeuralNetwork {
  constructor(insightApi: InsightApi, response: NeuralNetwork) {
    super(insightApi, response);
  }

  get id(): string { return this.response.id; }

  get totalError(): number { return this.response.totalError; }

  get inputCount(): number { return this.response.inputCount; }

  get outputCount(): number { return this.response.outputCount; }

  get layerCount(): number { return this.response.layerCount; }

  get parameters(): ParameterSetMap[] { return this.response.parameters; }

  forwardPass(inputs: string[] | number[]): ng.IPromise<void> {
    return this.postRequestProcessing(
        this.insightApi.networkCommand(this.id, "forward_pass", toNumbers(inputs)));
  }

  backwardPass(expected: number[]): ng.IPromise<void> {
    return this.postRequestProcessing(
        this.insightApi.networkCommand(this.id, "backward_pass", toNumbers(expected)));
  }
}

class TrainerDomain extends DomainObject<Trainer> implements Trainer {
  constructor(insightApi: InsightApi, response: Trainer) { super(insightApi, response); }

  get id(): string { return this.response.id; }

  get networkId(): string { return this.response.networkId; }

  get batchSize(): number { return this.response.batchSize; }

  get stepTally(): number { return this.response.stepTally; }

  get batchTally(): number { return this.response.batchTally; }

  get batchResults(): TrainerBatchResult[] {
    return this.response.batchResults;
  }

  singleTrain(): ng.IPromise<void> {
    return this.postRequestProcessing(
        this.insightApi.trainerCommand(this.id, "single_train"));
  }

  batchTrain(batchSize: string | number = -1): ng.IPromise<void> {
    return this.postRequestProcessing(
        this.insightApi.trainerCommand(this.id, "batch_train", toNumber(batchSize)));
  }
}

enum NetworkType {
  STANDARD_FEED_FORWARD,
  QUADRATIC_FEED_FORWARD
}

enum TrainerType {
  CLOSED_FORM_FUNCTION
}

type NetworkCommand = 'forward_pass' | 'backward_pass' | 'adjust_weights' |
                      'adjust_biases' | 'adjust_parameters';

type TrainerCommand = 'batch_train' | 'single_train';

class InsightApi {

  constructor(private $http: ng.IHttpService) {}

  createNetwork(layers: string[] | number[], type: NetworkType,
                options: Object): ng.IPromise<NeuralNetworkDomain> {
    return this
        .postRequestProcessing(this.$http.post(
            "/create_network",
            {layers : toNumbers(layers), type : NetworkType[type], options}))
        .then(result => new NeuralNetworkDomain(this, result));
  }

  createTrainer(network: NeuralNetwork, type: TrainerType,
                options: Object): ng.IPromise<TrainerDomain> {
    return this.postRequestProcessing(this.$http.post(
        "/create_trainer",
        {networkId : network.id, type : TrainerType[type], options}))
        .then(result => new TrainerDomain(this, result));
  }

  networkCommand(networkId: string, command: NetworkCommand,
                 ...args: any[]): ng.IPromise<NeuralNetwork> {
    return this.remoteCommand(networkId, command, args);
  }

  trainerCommand(trainerId: string, command: TrainerCommand,
                 ...args: any[]): ng.IPromise<Trainer> {
    return this.remoteCommand(trainerId, command, args);
  }

  private remoteCommand(targetId: string, command: string,
                        ...args: any[]): ng.IPromise<any> {
    return this.postRequestProcessing(
        this.$http.post(`/remote_command/${targetId}/${command}`, {args}));
  }

  private postRequestProcessing(response: ng.IPromise<any>) {
    return response.then(r => r.data);
  }
}

class InsightController {
  private feedForward: NeuralNetworkDomain;

  constructor(private insightApi: InsightApi) {}
}

class TrainingConsole {
  networkType: NetworkType = NetworkType.QUADRATIC_FEED_FORWARD;

  private network: NeuralNetworkDomain|null = null;
  private trainer: TrainerDomain|null = null;

  constructor(private insightApi: InsightApi) { }

  get hasNetwork(): boolean {
    return this.network != null;
  }

  get hasTrainer(): boolean {
    return this.trainer != null;
  }

  createNetwork(layers: number[]) {
    this.insightApi.createNetwork(layers, this.networkType,)
  }
}


let insight = angular.module('insight', [ 'ngMaterial' ])
                  .service('insightApi', InsightApi)
                  .controller('insightController', InsightController);

angular.bootstrap(document, [ insight.name ]);
