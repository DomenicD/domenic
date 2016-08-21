import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {NetworkType} from "../../../../common/service/api/insight-api-message";
import {PolymerElement} from "@vaadin/angular2-polymer";
import {MdInput} from "@angular2-material/input";

@Component({
  moduleId: module.id,
  selector: 'app-create-network',
  templateUrl: 'create-network.component.html',
  styleUrls: ['create-network.component.css'],
  encapsulation: ViewEncapsulation.Native,
  directives: [
    PolymerElement('paper-dropdown-menu'),
    PolymerElement('paper-listbox'),
    PolymerElement('paper-item'),
    PolymerElement('paper-input'),
    MdInput
  ]
})
export class CreateNetworkComponent implements OnInit {

  layers: string = "";
  networkTypes: string[];
  networkTypeIndex: number;


  constructor() {
    this.networkTypes = Object.keys(NetworkType).filter(key => Number.isNaN(Number(key)));
    this.networkType = NetworkType.QUADRATIC_FEED_FORWARD;
  }

  ngOnInit() {
  }

  get networkType(): NetworkType {
    return NetworkType[this.networkTypes[this.networkTypeIndex]];
  }

  set networkType(type: NetworkType) {
    this.networkTypeIndex = this.networkTypes.indexOf(NetworkType[type]);
  }
}
