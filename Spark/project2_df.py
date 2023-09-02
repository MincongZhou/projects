from pyspark import SparkContext, SparkConf
from pyspark.sql import  SparkSession
from operator import add
import math
from pyspark.sql import Row, Column
from pyspark.sql.functions import *

class Project2_df:
    def run(self, inputPath, outputPath, stopwordsPath, k):
        spark = SparkSession.builder.master("local").appName("project2_df").getOrCreate()
        fileDF = spark.read.text(inputPath).withColumnRenamed("value", "raw_data")
        swDF = spark.read.text(stopwordsPath).withColumnRenamed("value", "stopwords")
        # stopwords list
        sw_list = swDF.rdd.map(lambda x: x[0]).collect()
        # extract year from raw data
        extract_year = fileDF.withColumn("Year", split(fileDF["raw_data"], ",")[0][0:4])
        # convert headlines to list
        convert_headlines = extract_year.withColumn("Headline", split(split(extract_year["raw_data"], ",")[1], " "))
        # drop raw_data column
        drop_raw = convert_headlines.drop(col("raw_data"))
        # remove repeated in one headline
        rm_repeated = drop_raw.withColumn("Headline", array_distinct("Headline"))
        # explode to term
        explode_df = rm_repeated.withColumn("Headline", explode("Headline")).withColumnRenamed("Headline", "Term")
        # remove stopwords
        rm_sw = explode_df.filter(~explode_df["Term"].isin(sw_list))
        # count number of years
        numOfYears = rm_sw.select("Year").distinct().count()
        # compute TF
        tf = rm_sw.groupBy("Term", "Year").count().withColumnRenamed("count", "TF")
        # compute IDF
        idf = rm_sw.groupBy("Term").agg(countDistinct("Year").alias("year_freq"))
        # join tf and idf
        join_df = tf.join(idf, on="Term")
        # compute weight
        weight = join_df.select("Term", "Year",
                                (round(join_df["TF"] * log10(numOfYears / join_df["year_freq"]), 6)).alias("weight"))
        # sort
        sort_df = weight.orderBy(["Year", "weight", "Term"], ascending=[1, 0, 1])

        # modified output
        res_df = sort_df.groupBy("Year").agg(slice(collect_list(array("Year", "Term", "weight")), 1, k).alias("output")).orderBy("Year")
        def output_func(x):
            (year, l) = x
            tmp = []
            for e in l:
                str1 = f"{e[1]},{e[2]}"
                tmp.append(str1)
            str2 = f"{year}\t" + ";".join(tmp)
            return str2
        output_rdd = res_df.rdd.map(output_func)

        # save to output file
        #output_rdd.foreach(print)
        output_rdd.saveAsTextFile(outputPath)





if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Wrong inputs")
        sys.exit(-1)
    Project2_df().run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
