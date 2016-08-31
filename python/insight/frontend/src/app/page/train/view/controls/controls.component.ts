import {Component, OnInit, Input, Output, EventEmitter, ViewEncapsulation} from '@angular/core';
import {PolymerElement} from "@vaadin/angular2-polymer";
import {TrainerDomain} from "../../../../common/domain/trainer";

@Component({
  moduleId: module.id,
  selector: 'app-controls',
  templateUrl: 'controls.component.html',
  styleUrls: ['controls.component.css'],
  directives: [
    PolymerElement('paper-card'),
    PolymerElement('paper-input'),
    PolymerElement('paper-button')
  ],
  encapsulation : ViewEncapsulation.Native
})
export class ControlsComponent implements OnInit {

  @Input()
  trainer: TrainerDomain = null;
  isTraining: boolean = false;

  constructor() { }

  ngOnInit() {
  }

  train(batchSize: number = 1) {
    this.isTraining = true;
    let trainingPromise = batchSize > 1 ?
      this.trainer.batchTrain(batchSize) :
      this.trainer.singleTrain();
    trainingPromise.then(_ => this.isTraining = false, _ => this.isTraining = false);
  }
}
