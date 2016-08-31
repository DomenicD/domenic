import {Component, OnInit, Input, ViewEncapsulation} from '@angular/core';
import {ArrayObservable} from "rxjs/observable/ArrayObservable";
import {Observable, Subscription} from "rxjs/Rx";
import {
  ParameterSet,
  ParameterSetMap,
  TrainerBatchResult
} from "../../../../common/service/api/insight-api-message";
import {TrainerDomain} from "../../../../common/domain/trainer";
import {PolymerElement} from "@vaadin/angular2-polymer";

class ParameterDataPoint {
  constructor(public value: number, public gradient: number,
              public delta: number) {}
}

class ParameterEvolution {
  dataPoints: ParameterDataPoint[] = [];
  constructor(public name: string) {}
}

@Component({
  moduleId : module.id,
  selector : 'app-details',
  templateUrl : 'details.component.html',
  styleUrls : [ 'details.component.css' ],
  directives: [
    PolymerElement('vaadin-grid')
  ],
  encapsulation : ViewEncapsulation.Native
})
export class DetailsComponent implements OnInit {

  cachedParameterEvolutions: ParameterEvolution[];
  parameterEvolutions: Map<string, ParameterEvolution> =
      new Map<string, ParameterEvolution>();

  private _trainer: TrainerDomain = null;
  private batchResultSubscription: Subscription = null;

  constructor() {}

  ngOnInit() {}

  @Input()
  get trainer(): TrainerDomain {
    return this._trainer;
  }

  set trainer(value: TrainerDomain) {
    this._trainer = value;
    if (this.batchResultSubscription != null) {
      this.batchResultSubscription.unsubscribe();
    }
    this.batchResultSubscription = this.trainer.onBatchResult.subscribe(
        (batchResult: TrainerBatchResult) =>
            this.updateParameters(batchResult));
  }

  private paramValues(paramSet: ParameterSet): Observable<number[]> {
    return Observable.zip(
        ArrayObservable.create([].concat(...paramSet.values)),
        ArrayObservable.create([].concat(...paramSet.gradients)),
        ArrayObservable.create([].concat(...paramSet.deltas)));
  }

  private updateParameters(batchResult: TrainerBatchResult) {
    let paramSetMaps = batchResult.parameters;
    for (let paramSetMap of paramSetMaps) {
      for (let key in paramSetMap) {
        let paramSet = paramSetMap[key];
        this.paramValues(paramSet)
            .map((tuple: [ number, number, number ], index: number) => {
              let name = `${paramSet.name}_${index}`;
              if (!this.parameterEvolutions.has(name)) {
                this.parameterEvolutions.set(name,
                                             new ParameterEvolution(name));
              }
              this.parameterEvolutions.get(name).dataPoints.unshift(
                  new ParameterDataPoint(tuple[0], tuple[1], tuple[2]));
            })
            .subscribe(_ => this.onParameterUpdateCompleted());
      }
    }
  }

  private onParameterUpdateCompleted() {
    this.cachedParameterEvolutions =
        Array.from(this.parameterEvolutions.values());
  }
}
