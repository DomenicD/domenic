import {NgModule, CUSTOM_ELEMENTS_SCHEMA} from '@angular/core';
import {TrainComponent} from "./train.component";
import {CreateNetworkComponent} from "./view/create-network/create-network.component";
import {CommonModule} from "@angular/common";
import {MdInputModule} from "@angular2-material/input";

@NgModule({
  imports : [ CommonModule, MdInputModule ],
  providers : [],
  declarations : [ TrainComponent, CreateNetworkComponent ],
  exports : [ TrainComponent ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA]
})
export class TrainModule {
}
