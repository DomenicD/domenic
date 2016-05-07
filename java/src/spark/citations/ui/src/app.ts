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

    constructor() {
        this.citations = data;
        this.citationIds = Object.keys(data).map(Number);
        this.citationId = this.citationIds[0];
    }

    get citationId(): number {
        return this._citationId;
    }

    set citationId(value: number) {
        if (value != this.citationId) {
            this._citationId = value;
            this.citation = this.citations[this.citationId];
            this.createForceGraph();
        }
    }

    // TODO(domenicd): Use focus point so pull the depths apart.
    createForceGraph() {
        let nodeMap: {[key: number]: CitationNode} = {};
        nodeMap[this.citation.id] = CitationNode.emptyNode(this.citation.id);
        let maxCount = 0;
        let maxDepth = 0;
        let maxQuality = 0;
        let nodesAtDepth: {[depth: number]: number} = {};

        for (let rank of this.citation.ranks) {
            nodeMap[rank.id] = new CitationNode(rank);
            // Compute global attributes.
            if (rank.count > maxCount) maxCount = rank.count;
            if (rank.depth > maxDepth) maxDepth = rank.depth;
            if (rank.quality > maxQuality) maxQuality = rank.quality;
            if (!nodesAtDepth[rank.depth]) {
                nodesAtDepth[rank.depth] = 1;
            } else {
                nodesAtDepth[rank.depth]++;
            }

            for (let id of rank.citations) {
                if (!nodeMap[id]) {
                    nodeMap[id] = CitationNode.emptyNode(id);
                }
                nodeMap[id].children.push(nodeMap[rank.id]);
            }
        }

        let citationLinks: CitationLink[] = [];

        for (let rank of this.citation.ranks) {
            let rankNode = nodeMap[rank.id];
            for (let sourceId of rank.citations) {
                citationLinks.push(new CitationLink(nodeMap[sourceId], rankNode))
            }
        }

        let citationNodes: CitationNode[] = [];

        let depthIndices: number[] = [];

        for (let i = 0; i <= maxDepth; i++) {
            depthIndices.push(0);
        }

        for (let id of Object.keys(nodeMap)) {
            var node = nodeMap[Number(id)];
            node.maxCount = maxCount;
            if (node.depthIndex === -1) {
                node.depthIndex = depthIndices[node.depth]++;
            }
            citationNodes.push(node);
        }

        let yScale = d3.scale.ordinal()
            .domain(<any>(d3.range(maxDepth + 1)))
            .rangeRoundPoints([0, D3_HEIGHT], .5);

        // TODO(domenicd): Use nodesAtDepth to create a scale for each
        let xScales = depthIndices.map(d =>
            d3.scale.linear()
                .domain([-1, d])
                .range([0, D3_WIDTH]));

        let colorScale = d3.scale.linear().domain([0, maxQuality]).range(["blue", "red"]);

        let onTick = this.getTickFn(
            (node: CitationNode) => xScales[node.depth](node.depthIndex),
            (node: CitationNode) => yScale(node.depth),
            (node: CitationNode) => colorScale(node.quality));


        if (this.svg) {            
            this.svg.remove();
        }

        this.svg = d3.select(D3_CONTAINER).append("svg")
            .attr("width", D3_WIDTH)
            .attr("height", D3_HEIGHT);

        this.links = this.svg.selectAll(".link")
            .data(citationLinks)
            .enter()
            .append("line")
            .attr("class", "link");

        this.nodes = this.svg.selectAll(".node")
            .data(citationNodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", (n) => 5 + (n.count/maxCount) * CitationNode.RADIUS);

        d3.layout.force()
            .nodes(citationNodes)
            .links(citationLinks)
            .size([D3_WIDTH, D3_HEIGHT])
            .charge(0)
            .linkStrength(0)
            .on("tick", onTick)
            .start();
    }

    getTickFn(xDomain: (node: CitationNode) => number,
              yDomain: (node: CitationNode) => number,
              color: (node: CitationNode) => string): Function {
        // TODO(domenicd): Can use link count to alter Y axis by a little.
        return () => {
            this.nodes
                .attr("cx", (n) => xDomain(n))
                .attr("cy", (n) => this.getY(yDomain, n))
                .style("fill", color);

            this.links
                .attr("x1", (l) => xDomain(l.source))
                .attr("y1", (l) => this.getY(yDomain, l.source))
                .attr("x2", (l) => xDomain(l.target))
                .attr("y2", (l) => this.getY(yDomain, l.target));
        };
    }

    getY(yDomain: (node: CitationNode) => number, node: CitationNode) {
        let y = yDomain(node);
        y += (node.count / node.maxCount) * -CitationNode.RADIUS;
        return y;
    }
}

class CitationNode implements d3.layout.force.Node {
    id: number;
    count: number;
    quality: number;
    depth: number;
    maxCount: number;
    depthIndex: number = -1;

    children: CitationNode[] = [];

    constructor(rank: CitationRank) {
        this.id = rank.id;
        this.count = rank.count;
        this.quality = rank.quality;
        this.depth = rank.depth;
    }

    static get RADIUS() { return 50; }

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