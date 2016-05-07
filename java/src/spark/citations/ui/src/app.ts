let app = angular.module("citationAtlas", ['ngMaterial', 'ngMessages']);

app.config(function($mdThemingProvider: ng.material.IThemingProvider) {
    $mdThemingProvider.theme('default')
        .primaryPalette('indigo')
        .accentPalette('pink');
});

interface Citation {
    id: number;
    ranks: CitationRank[];
}

interface CitationRank {
    id: number;
    count: number;
    quality: number;
    depth: number;
    citations: number[];
}

declare var data: {[key: number]: Citation};

const D3_HEIGHT = 700;
const D3_WIDTH = 1280;
const D3_CONTAINER = "#d3-content";

class CitationController {

    private _citationId: number;
    private citation: Citation;

    citationIds: number[];
    citations: {[key: number]: Citation};

    svg: d3.Selection<any>;
    links: d3.Selection<any>;
    nodes: d3.Selection<any>;
    force: d3.layout.Force<d3.layout.force.Link<d3.layout.force.Node>, d3.layout.force.Node>;
    constructor() {
        this.citations = data;
        this.citationIds = Object.keys(data).map(Number);
        this.citationId = 10102; //this.citationIds[0];
    }

    get citationId(): number {
        return this._citationId;
    }

    set citationId(value: number) {
        if (value != this.citationId) {
            this._citationId = value;
            this.citation = this.citations[this.citationId];
            this.updateGraph();
        }
    }

    onTick() {
    this.links.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    this.nodes.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
    }

    updateGraph() {
        let nodeMap: {[key: number]: CitationNode} = {};
        nodeMap[this.citation.id] = CitationNode.emptyNode(this.citation.id);
        let maxCount = 0;

        for (let rank of this.citation.ranks) {
            nodeMap[rank.id] = new CitationNode(rank);
            if (rank.count > maxCount) {
                maxCount = rank.count;
            }
            for (let id of rank.citations) {
                if (!nodeMap[id]) {
                    nodeMap[id] = CitationNode.emptyNode(id);
                }
            }
        }

        let links: CitationLink[] = [];

        for (let rank of this.citation.ranks) {
            let rankNode = nodeMap[rank.id];
            for (let sourceId of rank.citations) {
                links.push(new CitationLink(nodeMap[sourceId], rankNode))
            }
        }

        let nodes: CitationNode[] = [];        

        for (let id of Object.keys(nodeMap)) {
            nodes.push(nodeMap[Number(id)]);
        }

        if (this.svg) {
            this.svg.remove();
        }

        this.svg = d3.select(D3_CONTAINER).append("svg")
            .attr("width", D3_WIDTH)
            .attr("height", D3_HEIGHT);

        this.links = this.svg.selectAll(".link");
        this.nodes = this.svg.selectAll(".node");

        this.force = d3.layout.force()
            .size([D3_WIDTH, D3_HEIGHT])
            .on("tick", () => this.onTick());

        this.force
            .nodes(nodes)
            .links(links)
            //.gravity(0)
            //.linkDistance(D3_HEIGHT/6)
            .charge((n: CitationNode) => -100 * (n.count/maxCount) * CitationNode.RADIUS)
            .linkStrength(.1)
            .start();

        this.links = this.links.data(links)
            .enter()
            .append("line")
            .attr("class", "link");

        this.nodes = this.nodes.data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", (n) => 12 + (n.count/maxCount) * CitationNode.RADIUS);
    }
}

class CitationNode {
    id: number;
    count: number;
    quality: number;
    depth: number;
    
    r: number;

    constructor(rank: CitationRank) {
        this.id = rank.id;
        this.count = rank.count;
        this.quality = rank.quality;
        this.depth = rank.depth;
    }

    static get RADIUS() { return 30; }

    static emptyNode(id: number): CitationNode {
        return new CitationNode({
            id: id,
            count: 0,
            quality: 0,
            depth: 0,
            citations: []
        });
    }
}

class CitationLink {
    constructor(public source: CitationNode, public target: CitationNode) { }
}

app.controller('citationController', CitationController);