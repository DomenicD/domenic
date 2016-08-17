import {InsightApiService} from "../services/api/insight-api.service";
import {Observable, Subscription} from "rxjs";

export abstract class DomainObject<T> {
  constructor(protected insightApi: InsightApiService, protected response: T) {}

  protected postRequestProcessing(responsePromise: Observable<T>):
      Promise<void> {
    return responsePromise.do(response => {this.response = response}).toPromise;
  }
}
