import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {NeuralNetworkDomain} from "../../common/domain/neural-network";
import {TrainerDomain} from "../../common/domain/trainer";
import {nvD3} from "ng2-nvd3";
import {ArrayObservable} from "rxjs/observable/ArrayObservable";
import {
  ParameterSet,
  ParameterSetMap
} from "../../common/service/api/insight-api-message";
import {Observable} from "rxjs/Rx";

class ParameterDataPoint {
  constructor(public value: number, public gradient: number,
              public delta: number) {}
}

class ParameterEvolution {
  dataPoints: ParameterDataPoint[] = [];
  constructor(public name: string) {}
}

const noop = _ => _;

@Component({
  moduleId : module.id,
  selector : 'app-train',
  templateUrl : 'train.component.html',
  styleUrls : [ 'train.component.css' ],
  // directives: [nvD3],
  encapsulation : ViewEncapsulation.Native
})
export class TrainComponent implements OnInit {

  neuralNetwork: NeuralNetworkDomain = null;
  trainer: TrainerDomain = null;
  isTraining = false;
  cachedParameterEvolutions: ParameterEvolution[];
  parameterEvolutions: Map<string, ParameterEvolution> =
      new Map<string, ParameterEvolution>();

  // @ViewChild(nvD3)
  // private d3Charts: nvD3;

  constructor() {}

  ngOnInit() {}

  get hasTrainer(): boolean { return this.trainer != null; }

  get hasNeuralNetwork(): boolean { return this.neuralNetwork != null; }

  onTrainerCreated(trainer: TrainerDomain) { this.trainer = trainer; }

  onNetworkCreated(network: NeuralNetworkDomain) {
    this.neuralNetwork = network;
    this.updateParameters(this.neuralNetwork.parameters);
  }

  train(batchSize: number = 1) {
    this.isTraining = true;
    let observable = batchSize > 1 ? this.trainer.batchTrain(batchSize)
                                   : this.trainer.singleTrain();
    observable.subscribe(_ => this.updateParameters(
                             this.trainer.batchResults.slice(-1)[0].parameters),
                         noop, () => this.isTraining = false);
  }

  private paramValues(paramSet: ParameterSet): Observable<number[]> {
    return Observable.zip(
        ArrayObservable.create([].concat(...paramSet.values)),
        ArrayObservable.create([].concat(...paramSet.gradients)),
        ArrayObservable.create([].concat(...paramSet.deltas)));
  }

  private updateParameters(paramSetMaps: ParameterSetMap[]) {
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
              this.parameterEvolutions.get(name).dataPoints.push(
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
