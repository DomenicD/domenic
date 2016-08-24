import {Component, OnInit, ViewEncapsulation, Output, EventEmitter} from '@angular/core';
import {NetworkType} from "../../../../common/service/api/insight-api-message";
import {PolymerElement} from "@vaadin/angular2-polymer";
import {MdInput} from "@angular2-material/input";
import {InsightApiService} from "../../../../common/service/api/insight-api.service";
import {NeuralNetworkDomain} from "../../../../common/domain/neural-network";
import {toNumber} from "../../../../common/util/parse";
import {UiFriendlyEnum} from "../../../../common/domain/ui-friendly-enum";

@Component({
  moduleId: module.id,
  selector: 'app-create-network',
  templateUrl: 'create-network.component.html',
  styleUrls: ['create-network.component.css'],
  encapsulation: ViewEncapsulation.Native,
  directives: [
    PolymerElement('paper-card'),
    PolymerElement('paper-dropdown-menu'),
    PolymerElement('paper-listbox'),
    PolymerElement('paper-item'),
    PolymerElement('paper-input'),
    PolymerElement('paper-button'),
    MdInput
  ]
})
export class CreateNetworkComponent implements OnInit {

  private isCreating: boolean = false;
  networkType: UiFriendlyEnum<NetworkType> = new UiFriendlyEnum<NetworkType>(NetworkType);

  layers: string = "";
  paramUpdateRate: number;

  @Output() onCreated = new EventEmitter<NeuralNetworkDomain>();

  constructor(private api: InsightApiService) {
    this.networkType.value = NetworkType.QUADRATIC_FEED_FORWARD;
    this.paramUpdateRate = .5;
  }

  ngOnInit() {
  }

  get isCreateDisabled(): boolean {
    return !this.validateLayers() || this.isCreating;
  }

  validateLayers(): boolean {
    if (this.layers == null || this.layers === "") {
      return false;
    }
    let numbers = this.layers.split(",").map(Number);
    if (numbers.length < 2) {
      return false;
    }
    return !numbers.some(Number.isNaN);
  }

  create() {
    this.isCreating = true;
    this.api.createNetwork(this.layers.split(","), this.networkType.value, {
      paramUpdateRate: toNumber(this.paramUpdateRate)
    }).subscribe(network => {
      this.onCreated.emit(network)
    }, error => {
      this.isCreating = false;
    });
  }
}
