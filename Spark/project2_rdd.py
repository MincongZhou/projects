from pyspark import SparkContext, SparkConf
import math
import sys

class Project2_rdd:
    def run(self, inputPath, outputPath, stopwordsPath, k):
        conf = SparkConf().setAppName("project2_rdd")
        sc = SparkContext(conf=conf)
        input = sc.textFile(inputPath)
        # stopwords list
        stopwords = sc.textFile(stopwordsPath).collect()
        # separate date and headline
        date_headline = input.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
        # remove stopwords
        def remove_sw_func(x):
            tmp = []
            for i in x[1]:
                if i not in stopwords:
                    tmp.append(i)
            x = (x[0], tmp)
            return x

        rm_sw = date_headline.map(lambda x: (x[0], x[1].split())).map(remove_sw_func)
        # remove repeated words in each headline
        rm_rw = rm_sw.map(lambda x: (x[0], set(x[1])))
        # take year from date
        year_headline = rm_rw.map(lambda x: (x[0][0:4], x[1]))
        # transform (year, headline) to (year, term)
        def term_func(x):
            (year, l) = x
            tmp = []
            for i in l:
                if i:
                    tmp.append((year, i))
            return tmp
        year_term = year_headline.flatMap(term_func)
        # cache year_term rdd
        year_term.persist()
        # calculate number of years
        numOfYears = len(year_term.countByKey())
        # compute TF
        tf1 = year_term.groupByKey().mapValues(list)
        def count_termPerYear_func(x):
            (year, l) = x
            count_dict = {}
            for term in l:
                count_dict[term]=l.count(term)
            tmp = []
            for term, count in count_dict.items():
                tmp.append((term, (year, count)))
            return tmp
        tf_res = tf1.map(count_termPerYear_func).flatMap(lambda x: x)
        tf_res.persist()
        # compute IDF
        term_year = year_term.map(lambda x: (x[1], x[0]))
        # cache term_year rdd
        term_year.persist()

        idf1 = term_year.groupByKey().mapValues(set).mapValues(len)
        # cache idf1
        idf1.persist()
        # def compute_idf_func(x):
        #     (term, num) = x
        #     idf = math.log10(numOfYears/num)
        #     return (term, idf)
        # idf_res = idf1.map(compute_idf_func)

        # compute weight
        weight1 = tf_res.join(idf1)
        def weight_func(x):
            (term, l) = x
            (year, count) = l[0]
            idf = l[1]
            weight = round(count*math.log10(numOfYears/idf),6)
            return (int(year), (term,weight))
        weight_res = weight1.map(weight_func)
        weight_res.persist()

        # sort the rdd
        sort_rdd = weight_res.sortBy(lambda x: (x[0], -x[1][1], x[1][0]))

        # choose first k elements for each year
        def firstK_func(x):
            (year, l) = x
            tmp = l[:k]
            return (year, tmp)
        first_k = sort_rdd.groupByKey().mapValues(list).sortBy(lambda x: x[0]).map(firstK_func)

        # modified output format
        def output_func(x):
            (year, l) = x
            tmp = []
            for e in l:
                str1 = f"{e[0]},{e[1]}"
                tmp.append(str1)
            str2 = f"{year}\t" + ";".join(tmp)
            return str2
        res = first_k.map(output_func)
        #res.foreach(print)
        res.persist()
        res.saveAsTextFile(outputPath)
        sc.stop()


if __name__ == "__main__":
    #Project2_rdd().run("file:///home/comp9313/project2/tiny-doc.txt","file:///home/comp9313/project2/output", "file:///home/comp9313/project2/stopwords.txt", 3)
    if len(sys.argv) != 5:
        print("Wrong inputs")
        sys.exit(-1)
    # for i in range(1,5):
    #     print(sys.argv[i])
    Project2_rdd().run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))