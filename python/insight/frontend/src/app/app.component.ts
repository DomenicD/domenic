import {Component, ViewEncapsulation} from '@angular/core';
import {InsightApiService} from "./common/service/api/insight-api.service";

@Component({
  moduleId: module.id,
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.css'],
  encapsulation: ViewEncapsulation.Native,
  providers: [InsightApiService]
})
export class AppComponent {
  title = 'app works!';
}
