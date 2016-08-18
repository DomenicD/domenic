import {Observable} from "rxjs";
import {InsightApiService} from "../service/api/insight-api.service";

export abstract class DomainObject<T> {
  constructor(protected insightApi: InsightApiService, protected response: T) {}

  protected postRequestProcessing(responsePromise: Observable<T>):
      Observable<this> {
    return responsePromise.do(response => this.response = response).map(_ => this);
  }
}
