package spark.citations

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx.{Edge, Graph}
import org.apache.spark.rdd.RDD

object CitationAtlas {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Citation Atlas")
    val sc = new SparkContext(conf)

    val citations: RDD[List[Long]] = sc.textFile(
      "file:///home/djdonato/git/domenic/java/src/spark/citations/Cit-HepPh.txt", 1)
      .map(s => s.split("\\s"))
      .filter(strings => strings.length == 2)
      .map(strings => List(strings(0).toLong, strings(1).toLong))
      .cache()

    val vertices: RDD[(Long, Publication)] = citations
      .flatMap(pair => pair)
      .distinct
      .map(id => (id, new Publication(id)))

    val edges: RDD[Edge[CitationMetadata]] = citations.map(longs => new Edge(longs.head, longs(1),
      new CitationMetadata()))

    val graph: Graph[Publication, CitationMetadata] = Graph.apply(vertices, edges)

    val ranks = graph.pageRank(.001).vertices

    ranks.sortBy(r => r._2).take(10).foreach(r => println("Vertex: " + r._1 + " Rank: " + r._2))
  }
}

class CitationMetadata extends Serializable {}

class Publication(val id: Long) extends Serializable {}
