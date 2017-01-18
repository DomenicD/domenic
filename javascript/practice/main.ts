
interface ElementFactory<T> {
    get(templateClone: HTMLElement, data: T): HTMLElement;
}

interface DataFactory<T> {
    readonly maxIndex: number;
    get(index: number): T;
}

class InfinityList<T> extends HTMLElement {
    private dataFactory: DataFactory<T>;
    private templateContent: DocumentFragment;
    private listContainer: HTMLDivElement;
    private elementFactory: ElementFactory<T>|null = null;
    private firstIndex: number = 0;


    constructor() {
        super();
        const shadowRoot = this.attachShadow({mode: 'open'});
        shadowRoot.innerHTML =
            `<div class=list-container></div>
             <slot></slot>`;
        this.listContainer = this.querySelector('.list-container') as HTMLDivElement;

        let template = this.querySelector('template');
        if (template == null) {
            throw new Error('InfinityList requires a child template.');
        }
        this.templateContent = template.content;
    }

    setDataFactory(dataFactory: DataFactory<T>) {
        this.dataFactory = dataFactory;
        this._onChanged();
    }

    setElementFactory(elementFactory: ElementFactory<T>) {
        this.elementFactory = elementFactory;
        this._onChanged()
    }

    private _onChanged() {
        if (this.dataFactory == null || this.elementFactory == null) {
            return;
        }



    }
}

class ColorOption {
    constructor(public readonly red: number,
                public readonly green: number,
                public readonly blue: number) {
        this.validateColor(red);
        this.validateColor(green);
        this.validateColor(blue);
    }

    private validateColor(colorValue: number) {
        if (colorValue < 0 || colorValue > 255) {
            throw new Error(`Value ${colorValue} is out of the valid color range.`);
        }
    }
}

class ColorOptionElementFactory implements ElementFactory<ColorOption> {
    get(templateClone: HTMLElement, data: ColorOption): HTMLElement {
        const colorDiv = templateClone.querySelector('.color-option');
        if (!(colorDiv instanceof HTMLElement)) {
            throw new Error('Invalid template, .color-option not found.')
        }
        (colorDiv as HTMLElement).style.backgroundColor =
            `rgb(${data.red}, ${data.green}, ${data.blue})`;
        return colorDiv;
    }

}

class ColorOptionDataFactory implements DataFactory<ColorOption> {
    get maxIndex(): number {
        return 256 * 256 * 256;
    }

    get(index: number): ColorOption {
        return new ColorOption(
            index % 256,
            index % (256 * 256),
            index % (256 * 256 * 256));
    }

}

customElements.define('infinity-list', InfinityList);

let colorList = document.querySelector('#infinity-color-list') as InfinityList<ColorOption>;
colorList.setElementFactory(new ColorOptionElementFactory());
colorList.setDataFactory(new ColorOptionDataFactory());


