
fin1 = open("observedflow.ssf","r")

fin2 = open("Q.txt","r")

fout = open("Q.ssf", "w")

mod = fin2.readlines()

countlines = 0

for line in fin1:
    record = line.split()
    record[-1] = mod[countlines]
    fout.write("  ".join(record))
    countlines = countlines + 1

fout.close()

fin2.close()

fin1.close()
    


