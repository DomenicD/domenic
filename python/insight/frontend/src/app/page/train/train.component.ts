import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {NeuralNetworkDomain} from "../../common/domain/neural-network";
import {TrainerDomain} from "../../common/domain/trainer";

@Component({
  moduleId: module.id,
  selector: 'app-train',
  templateUrl: 'train.component.html',
  styleUrls: ['train.component.css'],
  encapsulation: ViewEncapsulation.Native
})
export class TrainComponent implements OnInit {

  neuralNetwork: NeuralNetworkDomain = null;
  trainer: TrainerDomain = null;


  constructor() { }

  ngOnInit() {
  }

  get hasTrainer(): boolean {
    return this.trainer != null;
  }

  get hasNeuralNetwork(): boolean {
    return this.neuralNetwork != null;
  }

  onTrainerCreated(trainer: TrainerDomain) {
    this.trainer = trainer;
  }

  onNetworkCreated(network: NeuralNetworkDomain) {
    this.neuralNetwork = network;
  }


}
