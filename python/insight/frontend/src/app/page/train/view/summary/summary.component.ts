import {Component, OnInit, ViewChildren, QueryList} from '@angular/core';
import {nvD3} from "ng2-nvd3/lib/ng2-nvd3";

declare var d3;

interface LineChartData {
  color: string;
  key: string;
  values: Point[];
}

interface Point {
  x: number;
  y: number;
}

const lineChartOptions = {
  height : 450,
  margin : {top : 20, right : 20, bottom : 40, left : 55},
  x : function(d: Point) { return d.x; },
  y : function(d: Point) { return d.y; },
  useInteractiveGuideline : true,
  xAxis : {axisLabel : 'Time (ms)'},
  yAxis : {
    axisLabel : 'Voltage (v)',
    tickFormat : function(d) { return d3.format('.02f')(d); },
    axisLabelDistance : -10
  }
};

@Component({
  moduleId : module.id,
  selector : 'app-summary',
  templateUrl : 'summary.component.html',
  styleUrls : [ 'summary.component.css' ],
  directives : [ nvD3 ]
})
export class SummaryComponent implements OnInit {

  currentBatchChartOptions = lineChartOptions;
  currentBatchChartData;
  priorBatchChartOptions = lineChartOptions;
  priorBatchChartData;

  @ViewChildren(nvD3) charts: QueryList<nvD3>;

  constructor() {}

  ngOnInit() {}
}
