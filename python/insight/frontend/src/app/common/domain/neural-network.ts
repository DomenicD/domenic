import {NeuralNetwork, ParameterSetMap} from "../service/api/insight-api-message";
import {InsightApiService} from "../service/api/insight-api.service";
import {toNumbers} from "../util/parse";
import {Observable} from "rxjs";

export class NeuralNetworkDomain implements NeuralNetwork {
  constructor(private insightApi: InsightApiService, private response: NeuralNetwork) {
  }

  get id(): string { return this.response.id; }

  get totalError(): number { return this.response.totalError; }

  get inputCount(): number { return this.response.inputCount; }

  get outputCount(): number { return this.response.outputCount; }

  get layerCount(): number { return this.response.layerCount; }

  get parameters(): ParameterSetMap[] { return this.response.parameters; }

  forwardPass(inputs: string[] | number[]): Observable<void> {
    return this.insightApi.networkCommand<void>(this.id, "forward_pass", toNumbers(inputs));
  }

  backwardPass(expected: number[]): Observable<void> {
    return this.insightApi.networkCommand<void>(this.id, "backward_pass", toNumbers(expected));
  }
}
