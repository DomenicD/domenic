import {
  Component,
  OnInit,
  ViewEncapsulation,
  Input,
  EventEmitter,
  Output
} from '@angular/core';
import {getDefault} from "../../util/collection";

export enum HeatMapMode {
  LOCAL,
  GLOBAL
}

const DEFAULT_HISTORY = 50;
const DEFAULT_MODE = HeatMapMode.LOCAL;
const DEFAULT_USE_LOG_SCALE = true;

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
  useLogScale: boolean = DEFAULT_USE_LOG_SCALE;
  visibleCells: HeatMapCell[] = [];

  private cells: HeatMapCell[] = [];
  private _count: number = 0;

  constructor(public name: string) {}

  get count(): number { return this._count; }

  addValue(value: number) {
    this.cells.unshift(new HeatMapCell(value));
    this._count++;
    if (this.cells.length > this.history) {
      let removed = this.cells.pop();
      this.updateLocalMinMax(value, removed.actualValue);
    } else {
      this.updateLocalMinMax(value);
    }
  }

  update() {
    this.visibleCells = this.cells.slice(0, this.history);
    for (let cell of this.visibleCells) {
      cell.relativeValue = this.scaleValue(cell.actualValue);
    }
  }

  private updateLocalMinMax(value: number, removed?: number) {
    if (removed == this.localMax) {
      this.localMax = Math.max(...this.cells.map(c => c.actualValue));
    } else if (value > this.localMax) {
      this.localMax = value;
    }

    if (removed == this.localMin) {
      this.localMin = Math.min(...this.cells.map(c => c.actualValue));
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
    if (this.useLogScale) {
      absMax = this.logScale(absMax);
      let logValue = this.logScale(value);
      value = value < 0 ? -logValue : logValue;
    }
    let scaled = absMax > 0 ? value / absMax : 0;
    if (Math.abs(scaled) > 1) {
      console.error("Scaled should be clamped between 1 and -1");
    }
    return scaled;
  }

  private logScale(value: number): number {
    return Math.log(1 + Math.abs(value));
  }
}

export class HeatMapGroup {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;
  private _useLogScale: boolean = DEFAULT_USE_LOG_SCALE;

  constructor(public name: string, public rows: HeatMapRow[] = []) {}

  get history(): number { return this._history; }

  set history(value: number) {
    for (let row of this.rows) {
      row.history = value;
    }
    this._history = value;
  }

  get mode(): HeatMapMode { return this._mode; }

  set mode(value: HeatMapMode) {
    for (let row of this.rows) {
      row.mode = value;
    }
    this._mode = value;
  }

  get useLogScale(): boolean { return this._useLogScale; }

  set useLogScale(value: boolean) {
    for (let row of this.rows) {
      row.useLogScale = value;
    }
    this._useLogScale = value;
  }

  getRow(rowName: string): HeatMapRow {
    let row = this.rows.find(r => r.name === rowName);
    if (row == null) {
      row = new HeatMapRow(rowName);
      this.rows.push(row);
    }
    return row;
  }

  setGlobalExtrema(rowKey: string, min: number, max: number) {
    this.getRow(rowKey).globalMin = min;
    this.getRow(rowKey).globalMax = max;
  }

  update() {
    for (let row of this.rows) {
      row.update();
    }
  }
}

export class HeatMap {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;
  private _useLogScale: boolean = DEFAULT_USE_LOG_SCALE;

  constructor(public groups: HeatMapGroup[] = []) {}

  get history(): number { return this._history; }

  set history(value: number) {
    for (let group of this.groups) {
      group.history = value;
    }
    this._history = value;
  }

  get mode(): HeatMapMode { return this._mode; }

  set mode(value: HeatMapMode) {
    for (let group of this.groups) {
      group.mode = value;
    }
    this._mode = value;
  }

  get useLogScale(): boolean { return this._useLogScale; }

  set useLogScale(value: boolean) {
    for (let group of this.groups) {
      group.useLogScale = value;
    }
    this._useLogScale = value;
  }

  getGroup(groupName: string): HeatMapGroup {
    let group = this.groups.find(r => r.name === groupName);
    if (group == null) {
      group = new HeatMapGroup(groupName);
      this.groups.push(group);
    }
    return group;
  }

  update() {
    let maxes = new Map<string, number>();
    let mins = new Map<string, number>();

    // Find the global min and max for each row type.
    for (let group of this.groups) {
      for (let row of group.rows) {
        let rowName = row.name;
        let max = getDefault(maxes, rowName, Number.NEGATIVE_INFINITY);
        let min = getDefault(mins, rowName, Number.POSITIVE_INFINITY);
        if (row.localMax > max) {
          maxes.set(rowName, row.localMax);
        }
        if (row.localMin < min) {
          mins.set(rowName, row.localMin);
        }
      }
    }

    // Set the global min and max for each row type.
    for (let group of this.groups) {
      for (let row of group.rows) {
        row.globalMax = maxes.get(row.name);
        row.globalMin = mins.get(row.name);
      }
      // Have the group perform its update now that all its rows have been
      // updated.
      group.update();
    }
  }
}

export class GroupColumnSelectionEvent {
  constructor(public groupName: string, public rowName, public column: number) {
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
  private _useLogScale: boolean = DEFAULT_USE_LOG_SCALE;

  selectedGroup: string;
  selectedColumn: number;

  @Output()
  onGroupColumnSelection: EventEmitter<GroupColumnSelectionEvent> =
      new EventEmitter<GroupColumnSelectionEvent>();

  @Input() heatMap: HeatMap;
  @Input() showGroupName: boolean = true;
  @Input() showGroupDivider: boolean = false;
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

  @Input()
  get useLogScale(): boolean {
    return this._useLogScale;
  }

  set useLogScale(value: boolean) {
    if (this.heatMap != null && this.useLogScale != value) {
      this.heatMap.useLogScale = value;
      this.heatMap.update();
      this._useLogScale = value;
    }
  }

  get columnCount(): number {
    if (this.heatMap != null && this.heatMap.groups.length < 1 ||
        this.heatMap.groups[0].rows.length < 1) {
      return 0;
    }
    return this.heatMap.groups[0].rows[0].count;
  }

  ngOnInit() {}

  showRow(row: HeatMapRow): boolean {
    return this.visibleRows.indexOf(row.name) > -1;
  }

  selectGroupColumn(groupName: string, rowName: string, column: number) {
    let event = new GroupColumnSelectionEvent(groupName, rowName, column);
    this.selectedGroup = event.groupName;
    this.selectedColumn = event.column;
    this.onGroupColumnSelection.emit(event);
  }

  getColor(cell: HeatMapCell, rowIndex: number) {
    let percent = Math.abs(cell.relativeValue);
    let color = HeatColor.ALL[rowIndex % HeatColor.ALL.length];
    return cell.relativeValue > 0 ? color.warm(percent) : color.cold(percent);
  }
}

type ColorFunction = (percent: number) => [number, number, number];

function baseColor(start: number, percent: number) {
  return start + ((1 - start) * percent);
}

class HeatColor {

  static PINK: HeatColor = new HeatColor(p => [p, baseColor(.965, p), 1],
                                         p => [baseColor(.882, p), p, 1]);

  static RED: HeatColor =
      new HeatColor(p => [p, baseColor(.321, p), 1], p => [1, p, p]);

  static ORANGE: HeatColor = new HeatColor(p => [baseColor(.259, p), 1, p],
                                           p => [1, baseColor(.49, p), p]);

  static YELLOW: HeatColor = new HeatColor(p => [p, 1, baseColor(.816, p)],
                                           p => [1, baseColor(.882, p), p]);

  static ALL: HeatColor[] =
      [ HeatColor.RED, HeatColor.PINK, HeatColor.YELLOW, HeatColor.ORANGE ];

  constructor(private c: ColorFunction, private w: ColorFunction) {}

  cold(percent: number): string { return this.rgb(this.c(1 - percent)); }

  warm(percent: number): string { return this.rgb(this.w(1 - percent)); }

  private rgb(colors: [ number, number, number ]) {
    let [red, green, blue] = colors.map(c => (c * 100).toFixed(0) + '%');
    return `rgb(${red}, ${green}, ${blue})`;
  }
}
