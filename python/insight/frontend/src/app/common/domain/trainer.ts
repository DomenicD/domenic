import {DomainObject} from "./domain-object";
import {Trainer, TrainerBatchResult} from "../service/api/insight-api-message";
import {InsightApiService} from "../service/api/insight-api.service";
import {toNumber} from "../util/parse";
import {Observer} from "rxjs";
import {noop} from "rxjs/util/noop";
import {EventEmitter} from "@angular/core";
import {NextObserver} from "rxjs/Observer";

export class TrainerDomain extends DomainObject<Trainer> implements Trainer {

  onBatchResult = new EventEmitter<TrainerBatchResult>();

  constructor(insightApi: InsightApiService, response: Trainer) {
    super(insightApi, response);
  }

  get id(): string { return this.response.id; }

  get networkId(): string { return this.response.networkId; }

  get batchSize(): number { return this.response.batchSize; }

  get stepTally(): number { return this.response.stepTally; }

  get batchTally(): number { return this.response.batchTally; }

  get batchResults(): TrainerBatchResult[] {
    return this.response.batchResults;
  }

  singleTrain(): Promise<TrainerDomain> {
    return this.postRequestProcessing(
      this.insightApi.trainerCommand(this.id, "single_train"))
      .do(_ => this.emitBatchResult())
      .toPromise();
  }

  batchTrain(batchSize: string | number = -1): Promise<TrainerDomain> {
    return this.postRequestProcessing(
        this.insightApi.trainerCommand(this.id, "batch_train", toNumber(batchSize)))
      .do(_ => this.emitBatchResult())
      .toPromise();
  }

  private emitBatchResult(): void {
    this.onBatchResult.emit(this.batchResults.slice(-1)[0]);
  }
}
