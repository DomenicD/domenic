package spark.citations

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object CitationAtlas {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Citation Atlas")
    val sc = new SparkContext(conf)

    val citations: RDD[List[Long]] = sc.textFile(
      "file:///home/djdonato/git/domenic/java/src/spark/citations/test_data.txt", 1)
      .map(s => s.split("\\s"))
      .filter(strings => strings.length == 2)
      .map(strings => List(strings(0).toLong, strings(1).toLong))
      .cache()

    val vertices: RDD[(Long, Publication)] = citations
      .flatMap(pair => pair)
      .distinct
      .map(id => (id, new Publication(id)))

    val edges: RDD[Edge[Boolean]] = citations.map(longs => new Edge(longs.head, longs(1), false))

    val graph: Graph[Publication, Boolean] = Graph.apply(vertices, edges)

    val referenceMap: Map[VertexId, Set[VertexId]] =
      graph.collectNeighborIds(EdgeDirection.Out).collect()
        .groupBy(_._1)
        .mapValues(_.flatMap(v => v._2).toSet)
        // need to .map(identity) because of scala bug
        // http://stackoverflow.com/questions/32900862/map-can-not-be-serializable-in-scala
        .map(identity)

    val metadata = graph.vertices.mapValues((id, publication) =>
      getReferenceRanking(referenceMap, publication))

    metadata.foreach(m => print(m))
  }

  def getReferenceRanking(refMap: Map[VertexId, Set[VertexId]],
                          publication: Publication): CitationMetadata = {
    val rootId = publication.id
    val ranks = new mutable.HashMap[VertexId, ReferenceRank]()
    ranks.put(rootId, new ReferenceRank(rootId, 0))

    val visited = new mutable.HashSet[VertexId]()
    visited.add(rootId)

    val queue = new mutable.Queue[VertexId]()
    val maxDepth = 10
    var depth = 1
    recordCitations(ranks, queue, depth, rootId, refMap.get(rootId).get)
    
    while (queue.nonEmpty && depth < maxDepth) {
      val currentId = queue.dequeue()
      if (!visited.contains(currentId)) {
        recordCitations(ranks, queue, depth, currentId, refMap.get(currentId).get)
        visited.add(currentId)
        depth += 1
      }
    }

    // Compute the citationQuality by taking the average of the top n Publications that cite this
    // Publication.
    ranks.values.foreach(rank => {
      val n = 3
      rank.citationQuality = rank.citations
        .map(vertexId => ranks.get(vertexId).get.citationCount)
        .toList.sorted(Ordering[Long].reverse)
        .take(n)
        .sum / n
    })

    ranks.remove(rootId)

    new CitationMetadata(publication,
      ranks.values.toList.sortBy(x => (-x.citationCount, -x.citationQuality, x.depth)))
  }

  def recordCitations(ranks: mutable.HashMap[VertexId, ReferenceRank],
                      queue: mutable.Queue[VertexId],
                      depth: Long,
                      parentId: VertexId,
                      childrenIds: Set[VertexId]): Unit = {
    childrenIds.foreach(id => {
      if (!ranks.contains(id)) {
        ranks.put(id, new ReferenceRank(id, depth))
      }
      val rank = ranks.get(id)
      rank.get.citationCount += 1
      rank.get.citations.add(parentId)
      queue.enqueue(id)
    })
  }
}

class Publication(val id: Long) extends Serializable { }

class CitationMetadata(val publication: Publication,
                       val referenceRanks: List[ReferenceRank]) extends Serializable { }

class ReferenceRank(val id: VertexId, val depth: Long) extends Serializable {
  var citationCount: Long = 0
  var citationQuality: Double = 0.0
  val citations: mutable.Set[VertexId] = new mutable.HashSet[VertexId]()
}
