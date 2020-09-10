import numpy as np

a = 1.0 + 5
print(a)

# list can contain anything in the list
course_581 = [["Tusday", "Thursday"], "JianBo", 581]
print(course_581[0])

project1 = ["dooly"]
course_581.extend(project1)
print(course_581)


# Dictionary list[] is not hashble but tuple is, (  )
Dic = {581 : "computervision"}
print(Dic[581])

# == check equality, is check reference equality
c = [1,2,3]
c0 = [0]
c1 = [0]

print(c0 is c1)
print(c0 == c1)



