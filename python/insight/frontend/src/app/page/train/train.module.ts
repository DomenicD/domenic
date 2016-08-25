import {NgModule, CUSTOM_ELEMENTS_SCHEMA} from '@angular/core';
import {TrainComponent} from "./train.component";
import {CreateNetworkComponent} from "./view/create-network/create-network.component";
import {CommonModule} from "@angular/common";
import {CreateTrainerComponent} from "./view/create-trainer/create-trainer.component";

@NgModule({
  imports : [ CommonModule ],
  declarations : [ TrainComponent, CreateNetworkComponent, CreateTrainerComponent ],
  exports : [ TrainComponent ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA]
})
export class TrainModule {
}
