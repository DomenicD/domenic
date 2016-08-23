import {NgModule, CUSTOM_ELEMENTS_SCHEMA} from '@angular/core';
import {TrainComponent} from "./train.component";
import {CreateNetworkComponent} from "./view/create-network/create-network.component";
import {CommonModule} from "@angular/common";
import {MdInputModule} from "@angular2-material/input";
import {CreateTrainerComponent} from "./view/create-trainer/create-trainer.component";

@NgModule({
  imports : [ CommonModule, MdInputModule ],
  declarations : [ TrainComponent, CreateNetworkComponent, CreateTrainerComponent ],
  exports : [ TrainComponent ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA]
})
export class TrainModule {
}
