import {
  Component, OnInit, ViewChild, AfterViewInit, ViewEncapsulation,
  ElementRef
} from '@angular/core';
import {models} from 'nvd3';
import {format} from "d3";
import {select} from "d3";
import {addGraph} from "nvd3";

interface LineChartData {
  color: string;
  key: string;
  values: Point[];
}

interface Point {
  x: number;
  y: number;
}

/**************************************
 * Simple test data generator
 */
function sinAndCos() {
  var sin = [], sin2 = [], cos = [];

  // Data is represented as an array of {x,y} pairs.
  for (var i = 0; i < 100; i++) {
    sin.push({x : i, y : Math.sin(i / 10)});
    sin2.push({x : i, y : Math.sin(i / 10) * 0.25 + 0.5});
    cos.push({x : i, y : .5 * Math.cos(i / 10)});
  }

  // Line chart data should be sent as an array of series objects.
  return [
    {
      values : sin,      // values - represents the array of {x,y} data points
      key : 'Sine Wave', // key  - the name of the series.
      color : '#ff7f0e'  // color - optional: choose your own line color.
    },
    {values : cos, key : 'Cosine Wave', color : '#2ca02c'}, {
      values : sin2,
      key : 'Another sine wave',
      color : '#7777ff',
      area : true // area - set to true if you want this line to turn into a
                  // filled area chart.
    }
  ];
}

@Component({
  moduleId : module.id,
  selector : 'app-summary',
  templateUrl : 'summary.component.html',
  styleUrls : [ 'summary.component.css' ],
  encapsulation : ViewEncapsulation.Native
})
export class SummaryComponent implements OnInit, AfterViewInit {
  @ViewChild("currentChart") currentChart: ElementRef;

  @ViewChild("priorChart") priorChart: ElementRef;

  constructor() {}

  ngOnInit() {
  }

  ngAfterViewInit(): void {
    addGraph(() => {

      let chart =
          models.lineChart()
              .margin({
                left : 100
              }) // Adjust chart margins to give the x-axis some breathing room.
              .useInteractiveGuideline(
                  true) // We want nice looking tooltips and a guideline!
              .showLegend(
                  true)        // Show the legend, allowing users to turn on/off
                               // line series.
              .showYAxis(true) // Show the y-axis
              .showXAxis(true); // Show the x-axis

      chart
          .xAxis // Chart x-axis settings
          .axisLabel('Time (ms)')
          .tickFormat(format(',r'));

      chart
          .yAxis // Chart y-axis settings
          .axisLabel('Voltage (v)')
          .tickFormat(format('.02f'));

      /* Done setting the chart up? Time to render it!*/
      var myData = sinAndCos(); // You need data...
      select(this.currentChart.nativeElement).datum(myData).call(chart);

      return chart;

    });
  }
}
