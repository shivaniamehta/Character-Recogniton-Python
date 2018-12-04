import csv
import os

File1 = 'shuffle.csv'
File2 = 'train1.csv'
File3 = 'test1.csv'

with open(File1, "r") as r, open(File2, "a") as w:
     reader = csv.reader(r, lineterminator = "\n")
     writer = csv.writer(w, lineterminator = "\n")

     for counter,row in enumerate(reader):
         if counter<0: continue
         if counter>60000:break
         writer.writerow(row)
     r.close()

