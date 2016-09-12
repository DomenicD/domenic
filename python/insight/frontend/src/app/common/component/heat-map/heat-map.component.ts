import {Component, OnInit, ViewEncapsulation, Input} from '@angular/core';

export enum HeatMapMode {
  LOCAL,
  GROUP,
  GLOBAL
}

export class HeatMap {
  globalMax: number = 0;
  globalMin: number = 0;
  groupMax: number = 0;
  groupMin: number = 0;
  mode: HeatMapMode = HeatMapMode.LOCAL;
  scaledValues: number[][];
  values: number[][];

  private rowMax: Map<number, number> = new Map<number, number>();
  private rowMin: Map<number, number> = new Map<number, number>();

  constructor(public name: string, public history: number, rows: number) {
    this.values = [];
    for (let i = 0; i < rows; i++) {
      this.values.push([]);
      this.rowMax.set(i, 0);
      this.rowMin.set(i, 0);
    }
  }

  addValues(values: number[]) {
    let rowCount = this.values.length;
    if (values.length != rowCount) {
      throw new Error(
          `HeatMap expects ${rowCount} rows, ${values.length} were provided`);
    }
    for (let i = 0; i < rowCount; i++) {
      let value = values[i];
      let row = this.values[i];
      row.unshift(value);
      let removed: number;
      if (row.length > this.history) {
        removed = row.pop();
      }
      this.updateRowMinMax(value, i, removed);
    }
    this.updateGroupMinMax();
  }

  update() {
    this.scaledValues = this.values.map(
        (row, rowIndex) => row.map(value => this.scaleValue(value, rowIndex)))
  }

  private updateGroupMinMax() {
    this.groupMax = Math.max(...Array.from(this.rowMax.values()));
    this.groupMin = Math.min(...Array.from(this.rowMin.values()));
  }

  private updateRowMinMax(value: number, i: number, removed: number) {
    let max = this.rowMax.get(i);
    if (removed == max) {
      this.rowMax.set(i, Math.max(...this.values[i]));
    } else if (value > max) {
      this.rowMax.set(i, value);
    }

    let min = this.rowMin.get(i);
    if (removed == min) {
      this.rowMin.set(i, Math.min(...this.values[i]));
    } else if (value < min) {
      this.rowMin.set(i, value);
    }
  }

  private scaleValue(value: number, rowIndex: number): number {
    let max = 0;
    let min = 0;
    switch (this.mode) {
    case HeatMapMode.LOCAL:
      max = this.rowMax.get(rowIndex);
      min = this.rowMin.get(rowIndex);
      break;
    case HeatMapMode.GROUP:
      max = this.groupMax;
      min = this.groupMin;
      break;
    case HeatMapMode.GLOBAL:
      max = this.globalMax;
      min = this.globalMin;
      break;
    default:
      throw new Error(`HeatMapMode ${HeatMapMode[this.mode]} not implemented`);
    }
    let absMax = Math.max(Math.abs(max), Math.abs(min));
    let scaled = absMax > 0 ? value / absMax : 0;
    if (Math.abs(scaled) > 1) {
      console.error("Scaled should be clamped between 1 and -1");
    }
    return scaled;
  }
}

@Component({
  moduleId : module.id,
  selector : 'app-heat-map',
  templateUrl : 'heat-map.component.html',
  styleUrls : [ 'heat-map.component.css' ],
  encapsulation : ViewEncapsulation.Native
})
export class HeatMapComponent implements OnInit {

  @Input() heatMap: HeatMap;

  private _mode: HeatMapMode;

  constructor() {}

  @Input()
  get mode(): HeatMapMode {
    return this._mode;
  }

  set mode(value: HeatMapMode) {
    if (this.heatMap != null && this.mode != value) {
      this._mode = value;
      this.heatMap.mode = value;
      this.heatMap.update();
    }
  }

  ngOnInit() {}

  getColor(value: number) {
    let percent = ((1 - Math.abs(value)) * 100).toFixed(0) + "%";
    return value > 0 ? `rgb(100%, ${percent}, ${percent})` : `rgb(${percent}, ${percent}, 100%)`;
  }
}
