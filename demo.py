import os
import sys
import pandas
import numpy

input_file = sys.argv[1]


print(input_file)

def test_do(i):
    print("called test_do")
    with open(i, 'r') as inp:
        lines = inp.readlines()

    for line in lines:
        print(line.strip())


test_do(input_file)

class Animal:
 

    def one():
        print("called Animal.one")

    def two():
        print("called Animal.two")



#Animal.two()

def __main__():

    Animal.one()
    test_do()








