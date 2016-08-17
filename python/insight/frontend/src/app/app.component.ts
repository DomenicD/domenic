import { Component } from '@angular/core';
import {MdButton} from '@angular2-material/button';
import {MdCard} from "@angular2-material/card";

@Component({
  moduleId: module.id,
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.css'],
  directives: [
    MdButton,
    MdCard
  ]
})
export class AppComponent {
  title = 'app works!';
}
