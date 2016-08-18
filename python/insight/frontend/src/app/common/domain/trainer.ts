import {DomainObject} from "./domain-object";
import {Trainer, TrainerBatchResult} from "../service/api/insight-api-message";
import {InsightApiService} from "../service/api/insight-api.service";
import {Observable} from "rxjs";
import {toNumber} from "../util/parse";

export class TrainerDomain extends DomainObject<Trainer> implements Trainer {
  constructor(insightApi: InsightApiService, response: Trainer) { super(insightApi, response); }

  get id(): string { return this.response.id; }

  get networkId(): string { return this.response.networkId; }

  get batchSize(): number { return this.response.batchSize; }

  get stepTally(): number { return this.response.stepTally; }

  get batchTally(): number { return this.response.batchTally; }

  get batchResults(): TrainerBatchResult[] {
    return this.response.batchResults;
  }

  singleTrain(): Observable<TrainerDomain> {
    return this.postRequestProcessing(
        this.insightApi.trainerCommand(this.id, "single_train"));
  }

  batchTrain(batchSize: string | number = -1): Observable<TrainerDomain> {
    return this.postRequestProcessing(
        this.insightApi.trainerCommand(this.id, "batch_train", toNumber(batchSize)));
  }
}
