import pandas as pd

import csv

# read txt file
with open("CourseText.txt", 'r') as f:
    lines = f.readlines()

# create one column called department
courses = pd.DataFrame(lines, columns=["Department"])

# split department into department and ID by the -
courses[['Department', 'ID']] = courses['Department'].str.split("-", n=1, expand = True)

#split ID into ID and Title by the first space
courses[['ID', 'Title']] = courses['ID'].str.split(" ", n=1, expand = True)

# remove the \n in each column
courses['Title'] = courses['Title'].str.strip("\n")

print(courses)

#department is the ME for mech e and such, so course ID is after