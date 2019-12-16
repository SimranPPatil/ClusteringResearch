import sys
import subprocess

def execute_pipeline(fraction, filename):
    cmd = "python3 Library.py " + str(fraction) + " " + data_file + " " + label_file
    #cmd = "valgrind --tool=cachegrind python3 KMeans.py " + str(3) + " " + str(50) + " " + str(fraction) + " " + "make_moons_data.txt"
    with open(filename, "a+") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("enter data_file label_file")
        exit()

    data_file = sys.argv[1]
    label_file = sys.argv[2]
    
    filename = "log"
    fraction = 0.001
    execute_pipeline(fraction, filename)
    
    stride = 10
    for i in range(0, 100, stride):
        if i == 0:
            continue
        print(i*fraction)
        execute_pipeline(fraction*i, filename)

    stride *= 10
    for i in range(0, 1001, stride):
        if i == 0:
            continue
        print(i*fraction)
        execute_pipeline(fraction*i, filename)
