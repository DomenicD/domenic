import {Component, OnInit, ViewEncapsulation, Input} from '@angular/core';

export enum HeatMapMode {
  LOCAL,
  GLOBAL
}

const DEFAULT_HISTORY = 100;
const DEFAULT_MODE = HeatMapMode.LOCAL;

export class HeatMapCell {
  relativeValue: number = 0;

  constructor(public actualValue: number) {}
}

export class HeatMapRow {
  history: number = DEFAULT_HISTORY;
  globalMax: number = 0;
  globalMin: number = 0;
  localMax: number = 0;
  localMin: number = 0;
  mode: HeatMapMode = DEFAULT_MODE;
  visibleCells: HeatMapCell[] = [];

  private cells: HeatMapCell[] = [];

  constructor(public name: string) {}

  add(value: number) {
    this.cells.unshift(new HeatMapCell(value));
    let removed: number;
    if (this.cells.length > this.history) {
      removed = this.cells[this.history].actualValue;
    }
    this.updateLocalMinMax(value, removed);
  }

  update() {
    this.visibleCells = this.cells.slice(0, this.history);
    for (let cell of this.visibleCells) {
      cell.relativeValue = this.scaleValue(cell.actualValue);
    }
  }

  private updateLocalMinMax(value: number, removed: number) {
    if (removed == this.localMax) {
      this.localMax = Math.max(...this.cells);
    } else if (value > this.localMax) {
      this.localMax = value;
    }

    if (removed == this.localMin) {
      this.localMin = Math.min(...this.cells);
    } else if (value < this.localMin) {
      this.localMin = value;
    }
  }

  private scaleValue(value: number): number {
    let max = 0;
    let min = 0;
    switch (this.mode) {
    case HeatMapMode.LOCAL:
      max = this.localMax;
      min = this.localMin;
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

export class HeatMapGroup {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;
  private rowMap: Map<string, HeatMapRow>;

  constructor(public name: string, public rows: HeatMapRow[]) {
    this.rowMap = new Map<string, HeatMapRow>();
    for (let row of rows) {
      this.rowMap.set(row.name, row);
    }
  }

  get history(): number { return this._history; }

  set history(value: number) {
    for (let row of this.rowMap.values()) {
      row.history = value;
    }
    this._history = value;
  }

  get mode(): HeatMapMode { return this._mode; }

  set mode(value: HeatMapMode) {
    for (let row of this.rowMap.values()) {
      row.mode = value;
    }
    this._mode = value;
  }

  add(rowName: string, value: number) { this.rowMap.get(rowName).add(value); }

  setGlobalExtrema(rowKey: string, min: number, max: number) {
    this.rowMap.get(rowKey).globalMin = min;
    this.rowMap.get(rowKey).globalMax = max;
  }

  update() {
    for (let row of this.rowMap.values()) {
      row.update();
    }
  }
}

export class HeatMap {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;

  groupMap: Map<string, HeatMapGroup>;

  constructor(public groups: HeatMapGroup[]) {
    this.groupMap = new Map<string, HeatMapGroup>();
    for (let group of groups) {
      this.groupMap.set(group.name, group);
    }
  }

  get history(): number { return this._history; }

  set history(value: number) {
    for (let group of this.groupMap.values()) {
      group.history = value;
    }
    this._history = value;
  }

  get mode(): HeatMapMode { return this._mode; }

  set mode(value: HeatMapMode) {
    for (let group of this.groupMap.values()) {
      group.mode = value;
    }
    this._mode = value;
  }

  add(groupName: string, rowName: string, value: number) {
    this.groupMap.get(groupName).add(rowName, value);
  }

  setGlobalExtrema(rowName: string, min: number, max: number) {
    for (let group of this.groupMap.values()) {
      group.setGlobalExtrema(rowName, min, max);
    }
  }

  update() {
    for (let group of this.groupMap.values()) {
      group.update();
    }
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
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;

  @Input() heatMap: HeatMap;
  @Input() showGroupName: boolean = true;
  @Input() visibleRows: string[] = [];

  constructor() {}

  @Input()
  get history(): number {
    return this._history;
  }

  set history(value: number) {
    if (this.heatMap != null && this.history != value) {
      this.heatMap.history = value;
      this.heatMap.update();
      this._history = value;
    }
  }

  @Input()
  get mode(): HeatMapMode {
    return this._mode;
  }

  set mode(value: HeatMapMode) {
    if (this.heatMap != null && this.mode != value) {
      this.heatMap.mode = value;
      this.heatMap.update();
      this._mode = value;
    }
  }

  ngOnInit() {}

  showRow(name: string): boolean { return this.visibleRows.indexOf(name) > -1; }

  getColor(value: number) {
    let percent = ((1 - Math.abs(value)) * 100).toFixed(0) + "%";
    return value > 0 ? `rgb(100%, ${percent}, ${percent})`
                     : `rgb(${percent}, ${percent}, 100%)`;
  }
}
