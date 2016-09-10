import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {
  GoogleChart
} from "angular2-google-chart/directives/angular2-google-chart.directive";

@Component({
  moduleId : module.id,
  selector : 'app-summary',
  templateUrl : 'summary.component.html',
  styleUrls : [ 'summary.component.css' ],
  encapsulation : ViewEncapsulation.Native,
  directives : [ GoogleChart ]
})
export class SummaryComponent implements OnInit {
  line_ChartData = [
    [ 'Year', 'Sales', 'Expenses' ], [ '2004', 1000, 400 ],
    [ '2005', 1170, 460 ], [ '2006', 660, 1120 ], [ '2007', 1030, 540 ]
  ];

  line_ChartOptions = {
    title : 'Company Performance',
    curveType : 'function',
    legend : {position : 'bottom'}
  };

  constructor() {}

  ngOnInit() {}
}
