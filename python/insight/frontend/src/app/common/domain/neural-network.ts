export class NeuralNetworkDomain extends DomainObject<NeuralNetwork> implements
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
