import {Component, OnInit, EventEmitter, Output, Input} from '@angular/core';
import {TrainerDomain} from "../../../../common/domain/trainer";
import {NeuralNetworkDomain} from "../../../../common/domain/neural-network";
import {UiFriendlyEnum} from "../../../../common/domain/ui-friendly-enum";
import {TrainerType} from "../../../../common/service/api/insight-api-message";
import {InsightApiService} from "../../../../common/service/api/insight-api.service";

const LAMBDA_REGEX = /^lambda [a-z,]+: [ a-z0-9*+\-\/]+$/;

@Component({
  moduleId: module.id,
  selector: 'app-create-trainer',
  templateUrl: 'create-trainer.component.html',
  styleUrls: ['create-trainer.component.css']
})
export class CreateTrainerComponent implements OnInit {

  private isCreating: boolean = false;

  trainerType: UiFriendlyEnum<TrainerType> = new UiFriendlyEnum<TrainerType>(TrainerType);
  lambda: string = "lambda x: x**3 - x**2 + x + 1";
  domainMin: number = -10;
  domainMax: number = 10;
  batchSize: number = 100;

  @Input() network: NeuralNetworkDomain;
  @Output() onCreated = new EventEmitter<TrainerDomain>();

  constructor(private api: InsightApiService) {
    this.trainerType.value = TrainerType.CLOSED_FORM_FUNCTION;
  }

  ngOnInit() {
  }

  get isCreateDisabled(): boolean {
    return this.network == null || !this.validateLambda() || this.isCreating;
  }

  create() {
    this.isCreating = true;
    this.api.createTrainer(this.network, this.trainerType.value, {
      "function": this.lambda,
      "domain": [this.domainMin, this.domainMax],
      "batchSize": this.batchSize
    }).subscribe(trainer => this.onCreated.emit(trainer), e => this.isCreating = false);
  }

  private validateLambda(): boolean {
    if (this.lambda == null) {
      return false;
    }
    return LAMBDA_REGEX.test(this.lambda);
  }
}
