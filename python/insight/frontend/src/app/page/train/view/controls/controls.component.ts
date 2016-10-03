import {Component, OnInit, Input, ViewEncapsulation} from '@angular/core';
import {TrainerDomain} from "../../../../common/domain/trainer";
import {PolymerElement} from "@vaadin/angular2-polymer";
import {Observable} from "rxjs/Rx";

@Component({
  moduleId: module.id,
  selector: 'app-controls',
  templateUrl: 'controls.component.html',
  styleUrls: ['controls.component.css'],
  encapsulation : ViewEncapsulation.Native,
  directives: [PolymerElement('paper-icon-button')]
})
export class ControlsComponent implements OnInit {

  @Input()
  trainer: TrainerDomain;

  batchSize: number = 100;
  isTraining: boolean = false;
  trainingPromise: Promise<boolean>;

  private _isPlaying: boolean = false;

  constructor() { }

  get hasValidBatchSize(): boolean {
    return this.batchSize != null && this.batchSize > 0;
  }

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  ngOnInit() {
  }

  train() {
    this.isTraining = true;
    let trainingPromise = this.batchSize > 1 ?
      this.trainer.batchTrain(this.batchSize) :
      this.trainer.singleTrain();
    this.trainingPromise = trainingPromise
      .then(_ => this.isTraining = false, _ => this.isTraining = false);
  }

  play() {
    this._isPlaying = true;
    if (this.trainingPromise == null) {
      this.train();
    }
    this.trainingPromise = this.autoTrain();
  }

  private autoTrain(): Promise<boolean> {
    if (!this.isPlaying) {
      return;
    }
    return Observable.fromPromise(this.trainingPromise)
      .map(_ => this.train())
      .delay(1500)
      .toPromise()
      .then(_ => this.autoTrain());
  }

  pause() {
    this._isPlaying = false
  }
}
