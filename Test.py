import numpy

_lookup={"A": 1, "B": 2}

def convert(x):
    return x +"Test"

converters = {
0: convert, # in column 0 convert "A" --> 1, "B" --> 2,
# anything else to -1
}


if __name__ == "__main__":
# generate csv
    with open("tmp_sample.csv", "wb") as f:
        f.write("""A,1,this,67.8
B,2,should,56.7
C,3,be,34.5
A,4,skipped,12.3
""")

    # load csv
    a = numpy.loadtxt(
    "tmp_sample.csv",
    converters=converters,
    delimiter=",",
    usecols=(0, 1,3) # skip third column
    )
    print a
