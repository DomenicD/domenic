package spark.wordcounter;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.Arrays;

public class WordCounter {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Word Counter");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> textFile = sc.textFile("file:///home/djdonato/git/domenic/java/src/spark/wordcounter/test_file.txt", 1);
        JavaRDD<String> words = textFile.flatMap((FlatMapFunction<String, String>) s -> Arrays.asList(s.split(" ")));
        JavaPairRDD<String, Integer> pairs = words.mapToPair((PairFunction<String, String, Integer>) s -> new Tuple2<>(s, 1));
        JavaPairRDD<String, Integer> counts = pairs.reduceByKey((Function2<Integer, Integer, Integer>) (a, b) -> a + b);
        counts.saveAsTextFile("file:///home/djdonato/git/domenic/java/src/spark/wordcounter/out_test_file.txt");
    }
}
