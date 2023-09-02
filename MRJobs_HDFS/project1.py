from mrjob.job import MRJob
from mrjob.step import MRStep
import re


# ---------------------------------!!! Attention Please!!!------------------------------------
# Please add more details to the comments for each function. Clarifying the input
# and the output format would be better. It's helpful for tutors to review your code.

# Using multiple MRSteps is permitted, please name your functions properly for readability.

# We will test your code with the following comand:
# "python3 project1.py -r hadoop hdfs_input -o hdfs_output --jobconf mapreduce.job.reduces=2"

# Please make sure that your code can be compiled before submission.
# ---------------------------------!!! Attention Please!!!------------------------------------

class proj1(MRJob):
    # define your own mapreduce functions
    """------------------------------The first step------------------------------------------"""
    '''The fist step is mainly to calculate the probability'''

    # in-map combiner
    def mapper_init_1(self):
        self.tmp = {}

    # store Order-Inversion data in the tmp dictionary
    def mapper_1(self, _, line):
        info = re.split("[ *$&#/\t\n\f\"\'\\,.:;?!\[\](){}<>~\-_]", line.lower())[:2]
        freq = {}
        if len(info[0]) and len(info[1]):
            freq[info[1]] = freq.get(info[1], 0) + 1
            freq['*'] = freq.get('*', 0) + 1

        if info[0] in self.tmp:
            for k, v in freq.items():
                self.tmp[info[0]][k] = self.tmp[info[0]].get(k, 0) + int(v)
        else:
            self.tmp[info[0]] = freq

    # yield the key and value in the tmp dictionary using pairs
    def mapper_final_1(self):
        for user, location_dict in self.tmp.items():
            for location, v in location_dict.items():
                yield user + "," + location, v

    # calculate the  probability and yield
    def reducer_init_1(self):
        self.marginal = 0

    def reducer_1(self, key, values):
        user, location = key.split(",", 1)
        if location == "*":
            self.marginal = sum(values)
        else:
            count = sum(values)
            if self.marginal != 0:
                yield key, count / self.marginal

    '''------------------------------The second step------------------------------------------'''
    '''The second step is mainly to sort the data which is from the first step to satisfy the sorting requirement of 
    the project '''

    def mapper_2(self, key, value):
        info = key.split(",")
        user, location = info[0], info[1]
        yield location + "," + str(value) + "," + user, None

    def reducer_2(self, key, values):
        info = key.split(",")
        location, prob, user = info[0], info[1], info[2]
        yield location, user + "," + prob

    SORT_VALUES = True
    JOBCONF_1 = {
        # separate the key by ","
        'mapreduce.map.output.key.field.separator': ',',
        'partitioner': 'org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner',
        # sort the first column which is user ID in ascending order
        'mapreduce.partition.keypartitioner.options': '-k1,1',
        'mapreduce.job.output.key.comparator.class': 'org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator',
        # sort the second column which is location ID in ascending order
        'mapreduce.partition.keycomparator.options': '-k1,1 -k2,2'
    }

    JOBCONF_2 = {
        # separate the key by ","
        'mapreduce.map.output.key.field.separator': ',',
        'partitioner': 'org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner',
        # sort the first column which is location ID in ascending order
        'mapreduce.partition.keypartitioner.options': '-k1,1',
        'mapreduce.job.output.key.comparator.class': 'org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator',
        # sort the second column which is probability in descending order and sort the third column which is user ID
        # in ascending order
        'mapreduce.partition.keycomparator.options': '-k1,1 -k2,2nr -k3,3'
    }

    def steps(self):
        return [
            # you decide the number of steps used
            # first step
            MRStep(mapper_init=self.mapper_init_1, mapper=self.mapper_1, mapper_final=self.mapper_final_1,
                   reducer_init=self.reducer_init_1, reducer=self.reducer_1, jobconf=self.JOBCONF_1),
            # second step
            MRStep(mapper=self.mapper_2,
                   reducer=self.reducer_2,
                   jobconf=self.JOBCONF_2),

        ]


if __name__ == '__main__':
    proj1.run()
