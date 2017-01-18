class InfinityList extends HTMLElement {
    constructor() {
        super();
        this.elementFactory = null;
        this.firstIndex = 0;
        const shadowRoot = this.attachShadow({ mode: 'open' });
        shadowRoot.innerHTML =
            `<div class=list-container></div>
             <slot></slot>`;
        this.listContainer = this.querySelector('.list-container');
        let template = this.querySelector('template');
        if (template == null) {
            throw new Error('InfinityList requires a child template.');
        }
        this.templateContent = template.content;
    }
    setDataFactory(dataFactory) {
        this.dataFactory = dataFactory;
        this._onChanged();
    }
    setElementFactory(elementFactory) {
        this.elementFactory = elementFactory;
        this._onChanged();
    }
    _onChanged() {
        if (this.dataFactory == null || this.elementFactory == null) {
            return;
        }
    }
}
class ColorOption {
    constructor(red, green, blue) {
        this.red = red;
        this.green = green;
        this.blue = blue;
        this.validateColor(red);
        this.validateColor(green);
        this.validateColor(blue);
    }
    validateColor(colorValue) {
        if (colorValue < 0 || colorValue > 255) {
            throw new Error(`Value ${colorValue} is out of the valid color range.`);
        }
    }
}
class ColorOptionElementFactory {
    get(templateClone, data) {
        const colorDiv = templateClone.querySelector('.color-option');
        if (!(colorDiv instanceof HTMLElement)) {
            throw new Error('Invalid template, .color-option not found.');
        }
        colorDiv.style.backgroundColor =
            `rgb(${data.red}, ${data.green}, ${data.blue})`;
        return colorDiv;
    }
}
class ColorOptionDataFactory {
    get maxIndex() {
        return 256 * 256 * 256;
    }
    get(index) {
        return new ColorOption(index % 256, index % (256 * 256), index % (256 * 256 * 256));
    }
}
customElements.define('infinity-list', InfinityList);
let colorList = document.querySelector('#infinity-color-list');
colorList.setElementFactory(new ColorOptionElementFactory());
colorList.setDataFactory(new ColorOptionDataFactory());
