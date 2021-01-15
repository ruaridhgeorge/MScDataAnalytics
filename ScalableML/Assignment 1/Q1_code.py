### Question 1A ####################################################################### 

import pyspark
import re
from pyspark.sql.functions import regexp_extract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

#Read in data set
logFile=spark.read.text("ScalableML/Data/NASA_access_log_Jul95.gz").cache()

# Regex expressions to pre-process the data set into a data frame that is easier to manipulate
hostPattern=r'(^\S+\.[\S+\.]+\S+)\S'
datetimePattern=r'\[((\d{2}/\w{3}/\d{4})(:\d{2}:\d{2}:\d{2} -\d{4}))]'
timePattern=r'(\d{2}:\d{2}:\d{2}) '
method_uri=r'\"(\S+)\s(\S+)\s*(\S*)\"'
statusPattern=r'\s(\d{3})\s'
contentSizePattern=r'\s(\d+)$'

log_df=logFile.select(regexp_extract('value',hostPattern,1).alias('host'),
                       regexp_extract('value',datetimePattern,1).alias('dateTime'),
                       regexp_extract('value',timePattern,1).alias('time'),
                       regexp_extract('value',method_uri,1).alias('method'),
                       regexp_extract('value',method_uri,2).alias('endpoint'),
                       regexp_extract('value',method_uri,3).alias('protocol'),
                       regexp_extract('value',statusPattern,1).cast('integer').alias('status'),
                       regexp_extract('value',contentSizePattern,1).cast('integer').alias('content_size'))

#  Reference: https://opensource.com/article/19/5/log-data-apache-spark

# Drop null values from the data set: dropping nulls give different values
log_df = log_df.na.drop()

# Extracting each of the 6 time periods using regex
regex04 = '(0[0-3]):([0-5][0-9]):([0-5][0-9])'
regex08 = '(0[4-7]):([0-5][0-9]):([0-5][0-9])'
regex12 = '((0[8-9])|(1[0-1])):([0-5][0-9]):([0-5][0-9])'
regex16 = '(1[2-5]):([0-5][0-9]):([0-5][0-9])'
regex20 = '(1[6-9]):([0-5][0-9]):([0-5][0-9])'
regex24 = '(2[0-3]):([0-5][0-9]):([0-5][0-9])'

# 6 new data sets for each 4 hour period
logs04=log_df.filter(log_df.time.rlike(regex04))
logs08=log_df.filter(log_df.time.rlike(regex08))
logs12=log_df.filter(log_df.time.rlike(regex12))
logs16=log_df.filter(log_df.time.rlike(regex16))
logs20=log_df.filter(log_df.time.rlike(regex20))
logs24=log_df.filter(log_df.time.rlike(regex24))

# Days to average over. Note that days = 28 since data for the last 3 days of July is missing.
days = 28 

# Average amount of requests for each 4 hour period
avg04=logs04.count()/days
avg08=logs08.count()/days
avg12=logs12.count()/days
avg16=logs16.count()/days
avg20=logs20.count()/days
avg24=logs24.count()/days
avg_list = [avg04, avg08, avg12, avg16, avg20, avg24]

### Question 1B:###############################################################################

labels = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-24'] 
x = np.arange(len(labels))

plt.bar(x,avg_list)
plt.xticks(x, labels)
plt.xlabel('Time period')
plt.ylabel('Number of requests')
plt.title('Average number of requests by 4 hour period')
plt.savefig("/home/acq19rg/ScalableML/avg_list_bar.png")

for i in range(len(avg_list)):
  print('Average number of requests per day in July for 4-hour time block', i+1, 'is', avg_list[i], '\n')

### Question 1C: ###################################################################################

# .endswith and .contains gives two different counts, which would suggest .html appears earlier in a url.
log_html = log_df.filter(log_df.endpoint.endswith('.html')) # count=415971
log_html2 = log_df.filter(log_df['endpoint'].contains('.html')) # count=416078

tophtml = log_html.groupBy(['endpoint']).count().orderBy('count', ascending = False)
tophtml.show(20, False)
print('This table shows the top 20 requested html files in order of request count.')
tophtml

spark.stop()







