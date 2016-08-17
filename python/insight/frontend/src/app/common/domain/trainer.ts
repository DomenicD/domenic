export class TrainerDomain extends DomainObject<Trainer> implements Trainer {
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
