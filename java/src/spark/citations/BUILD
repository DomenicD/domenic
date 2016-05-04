load("@scala_rules//scala:scala.bzl", "scala_binary")

scala_binary(
    name = "citations",
    srcs = glob(["*.scala"]),
    deps = ["//third_party/java/spark"],
    main_class = "spark.citations.CitationAtlas"
)