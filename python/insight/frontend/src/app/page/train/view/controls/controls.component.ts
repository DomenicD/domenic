import {Component, OnInit, Input, ViewEncapsulation} from '@angular/core';
import {TrainerDomain} from "../../../../common/domain/trainer";

@Component({
  moduleId: module.id,
  selector: 'app-controls',
  templateUrl: 'controls.component.html',
  styleUrls: ['controls.component.css'],
  encapsulation : ViewEncapsulation.Native
})
export class ControlsComponent implements OnInit {

  @Input()
  trainer: TrainerDomain;
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
