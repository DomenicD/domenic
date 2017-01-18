import {Trainer, TrainerBatchResult, TrainerValidationResult} from "../service/api/insight-api-message";
import {InsightApiService} from "../service/api/insight-api.service";
import {toNumber} from "../util/parse";
import {EventEmitter} from "@angular/core";
import {Observable} from "rxjs/Rx";

export class TrainerDomain implements Trainer {

  onBatchResult = new EventEmitter<TrainerBatchResult>();

  constructor(private insightApi: InsightApiService, private response: Trainer) { }

  get id(): string { return this.response.id; }

  get networkId(): string { return this.response.networkId; }

  get batchSize(): number { return this.response.batchSize; }

  get stepTally(): number { return this.response.stepTally; }

  get batchTally(): number { return this.response.batchTally; }

  singleTrain(): Promise<TrainerDomain> {
    return this.insightApi.trainerCommand<TrainerBatchResult>(this.id, "single_train")
      .do(result => this.emitBatchResult(result))
      .map(_ => this)
      .toPromise();
  }

  batchTrain(batchSize: string | number, epochs: string | number): Promise<TrainerDomain> {
    return this.insightApi.trainerCommand<TrainerBatchResult>(this.id, "batch_train", toNumber(batchSize), toNumber(epochs))
      .do(result => this.emitBatchResult(result))
      .map(_ => this)
      .toPromise();
  }

  validate(): Observable<TrainerValidationResult> {
    return this.insightApi.trainerCommand<TrainerValidationResult>(this.id, "validate")
  }

  private emitBatchResult(result: TrainerBatchResult): void {
    this.onBatchResult.emit(result);
  }
}
