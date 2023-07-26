import csv
import time
import collections
from datetime import datetime


# Counting the number of patients who have a 12-hour gap.

patients = collections.defaultdict(list)

with open('etco2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if row[0] == 'Financial Number':
            continue
        if row[6].count(':') == 2:
            row[6] = row[6][:-3]
            tem1 = row[6].split(' ')
            tem2 = tem1[0].split('/')
            new_date_format = tem2[1] + '/' + tem2[2] + '/' + tem2[0][2:]
            row[6] = new_date_format + ' ' + tem1[1]
        patients[row[0]].append(time.mktime(datetime.strptime(row[6], '%m/%d/%y %H:%M').timetuple()))

cnt = 0
for p in patients:
    patients[p].sort()
    for i in range(1, len(patients[p])):
        if int(patients[p][i]) - int(patients[p][0]) > 43200:
            cnt += 1
            break
print(cnt)
print(len(patients))
print(cnt / len(patients))