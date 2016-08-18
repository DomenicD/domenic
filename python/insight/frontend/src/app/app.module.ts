import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';
import {AppComponent} from "./app.component";
import {MdMenuModule} from '@angular2-material/menu';
import {MdToolbarModule} from '@angular2-material/toolbar';
import {MdIconModule} from "@angular2-material/icon";
import {TrainModule} from "./page/train/train.module";

@NgModule({
  imports : [ BrowserModule, MdMenuModule, MdToolbarModule, MdIconModule, TrainModule ],
  providers : [],
  declarations : [ AppComponent ],
  exports : [ AppComponent ],
  bootstrap : [ AppComponent ]
})
export class AppModule {
}
