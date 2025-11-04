# area variables (in square meters )
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [[" hallway ", hall ],
         [" kitchen ", kit ],
         [" living room ", liv ],
         [" bedroom ", bed ],
         [" bathroom ", bath ]]
# Print out house
for room in house:
    print(f"Room: {room[0]}, Area: {room[1]} m2")

# Print out the type of house
print("Type of house:", type(house))