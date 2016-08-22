import {Component, OnInit, EventEmitter, Output, Input} from '@angular/core';
import {TrainerDomain} from "../../../../common/domain/trainer";
import {NeuralNetworkDomain} from "../../../../common/domain/neural-network";

@Component({
  moduleId: module.id,
  selector: 'app-create-trainer',
  templateUrl: 'create-trainer.component.html',
  styleUrls: ['create-trainer.component.css']
})
export class CreateTrainerComponent implements OnInit {

  @Input() network: NeuralNetworkDomain;
  @Output() onCreated = new EventEmitter<TrainerDomain>();

  constructor() { }

  ngOnInit() {
  }


}
