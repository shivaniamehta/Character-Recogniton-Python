import random
fid = open(r"C:\Users\Ashok Mehta\Downloads\handwritten_data_785.csv", "r")
li = fid.readlines()
fid.close()


random.shuffle(li)

fid = open("shuffle.csv", "w")
fid.writelines(li)
fid.close()