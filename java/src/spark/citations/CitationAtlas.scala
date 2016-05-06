package spark.citations

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

object CitationAtlas {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Citation Atlas")
    val sc = new SparkContext(conf)

    val citations: RDD[List[Long]] = sc.textFile(
      "file:///home/djdonato/git/domenic/java/src/spark/citations/Cit-HepPh.txt", 1)
      .map(s => s.split("\\s"))
      .filter(strings => strings.length == 2)
      .map(strings => List(strings(0).toLong, strings(1).toLong))

    val vertices: RDD[(Long, Publication)] = citations
      .flatMap(pair => pair)
      .distinct
      .map(id => (id, new Publication(id)))

    val edges: RDD[Edge[Boolean]] = citations.map(longs => new Edge(longs.head, longs(1), false))

    val graph: Graph[Publication, Boolean] = Graph.apply(vertices, edges).cache()

    val referenceMap: Map[VertexId, Set[VertexId]] =
      graph.collectNeighborIds(EdgeDirection.Out).collect()
        .groupBy(_._1)
        .mapValues(_.flatMap(v => v._2).toSet)
        // need to .map(identity) because of scala bug
        // http://stackoverflow.com/questions/32900862/map-can-not-be-serializable-in-scala
        .map(identity)

//    val metadata = graph.vertices.mapValues((id, publication) =>
//      getReferenceRanking(referenceMap, publication))
//
//    metadata
//      .persist(StorageLevel.MEMORY_AND_DISK)
//      .take(1)
//      .foreach(v => {
//        println(v._2)
//        println()
//      })

    println(getReferenceRanking(referenceMap, graph.vertices.first()._2))
  }

  def getReferenceRanking(refMap: Map[VertexId, Set[VertexId]],
                          publication: Publication): CitationMetadata = {
    val maxDepth = 2
    var depth = 1

    val rootId = publication.id
    val ranks = new mutable.HashMap[VertexId, ReferenceRank]()
    ranks.put(rootId, new ReferenceRank(rootId, 0))

    val visited = new mutable.HashSet[VertexId]()
    visited.add(rootId)
    val queue = new mutable.Queue[VertexId]()
        recordCitations(ranks, queue, rootId, refMap.get(rootId).get, maxDepth)

    var i = 0
    while (queue.nonEmpty && depth <= maxDepth) {
      val currentId = queue.dequeue()
      if (!visited.contains(currentId)) {
        visited.add(currentId)
        recordCitations(ranks, queue, currentId, refMap.get(currentId).get, maxDepth)
      }
      depth = ranks.get(currentId).get.depth

      if (i % 10 == 0) println(rootId + ": " + i)
      i += 1
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
                      parentId: VertexId,
                      childrenIds: Set[VertexId],
                      maxDepth: Int): Unit = {
    val depth = ranks.get(parentId).get.depth + 1
    childrenIds.foreach(id => {
      val hadRank: Boolean = ranks.contains(id)
      val isDepthAllowed: Boolean = depth <= maxDepth
      if (!hadRank && isDepthAllowed) {
        ranks.put(id, new ReferenceRank(id, depth))
      }
      if (hadRank || isDepthAllowed) {
        val rank = ranks.get(id)
        rank.get.citationCount += 1
        rank.get.citations.add(parentId)
        queue.enqueue(id)
      }
    })
  }
}

class Publication(val id: Long) extends Serializable { }

class CitationMetadata(val publication: Publication,
                       val referenceRanks: List[ReferenceRank]) extends Serializable {
  override def toString: String = {
    val sb: StringBuilder = new StringBuilder()
    sb.append("{\n")
    sb.append("id: ").append(publication.id).append(",\n")
    sb.append("ranks: [").append(referenceRanks.mkString(",\n")).append("]\n")
    sb.append("}")
    sb.toString()
  }
}

class ReferenceRank(val id: VertexId, val depth: Int) extends Serializable {
  var citationCount: Long = 0
  var citationQuality: Double = 0.0
  val citations: mutable.Set[VertexId] = new mutable.HashSet[VertexId]()

  override def toString: String = {
    val sb: StringBuilder = new StringBuilder()
    sb.append("{\n")
    sb.append("id: ").append(id).append(",\n")
    sb.append("count: ").append(citationCount).append(",\n")
    sb.append("quality: ").append(citationQuality).append(",\n")
    sb.append("depth: ").append(depth).append(",\n")
    sb.append("citations: [").append(citations.mkString(", ")).append("]\n")
    sb.append("}")
    sb.toString()
  }
}
