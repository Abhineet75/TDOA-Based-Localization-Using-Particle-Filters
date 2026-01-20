import random

POINTS = int(input("Enter the number of points: "))

inside_arc = 0

for shots in range(POINTS):

    x = random.random()
    y = random.random()

    if (x**2 + y**2 <= 1):
        inside_arc += 1

pi_calc = (inside_arc / POINTS) * 4

print("The value of PI is :", pi_calc)

