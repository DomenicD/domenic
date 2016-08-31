import {NgModule, CUSTOM_ELEMENTS_SCHEMA} from '@angular/core';
import {TrainComponent} from "./train.component";
import {
  CreateNetworkComponent
} from "./view/create-network/create-network.component";
import {CommonModule} from "@angular/common";
import {
  CreateTrainerComponent
} from "./view/create-trainer/create-trainer.component";
import {ControlsComponent} from "./view/controls/controls.component";
import {DetailsComponent} from "./view/details/details.component";
import {SummaryComponent} from "./view/summary/summary.component";

@NgModule({
  imports : [ CommonModule ],
  declarations : [
    TrainComponent, CreateNetworkComponent, CreateTrainerComponent,
    ControlsComponent, DetailsComponent, SummaryComponent
  ],
  exports : [ TrainComponent ],
  schemas : [ CUSTOM_ELEMENTS_SCHEMA ]
})
export class TrainModule {
}
