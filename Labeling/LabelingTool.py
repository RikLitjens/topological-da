import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

print(f"""
Welkom bij het TDA-Label programma slaaf euh ik bedoelde: gewaardeerde [DATA SCIENTIST]. 
Voer hieronder alstublieft uw naam in.
""")
username = input(f"Wat is je naam? ")
bag_number: int

if username.lower() == "thomas":
    bag_number = 0
elif username.lower() == "marco":
    bag_number = 1
elif username.lower() == "olivier":
    bag_number = 2
elif username.lower() == "alec":
    bag_number = 3
elif  username.lower() == "rik":
    bag_number = 4
else:
    print("Gebruikersnaam niet bekend in TDA-Label Programma"); exit()

print(f"""
Welkom [{username.upper()}]! 
Goed dat je er bent, je hebt bag {bag_number} aangewezen gekregen.
Veel plezier met labelen!""")
print()

this_folder = os.path.dirname(os.path.abspath(__file__))
csv_title = f"{username.lower()}_bag_{bag_number}.csv"

file = open(os.path.join(this_folder, csv_title), 'w')
file.write(f"Bag number, Image number, label\n")

print("""
Op het moment dat de afbeelding 1 connected component lijkt dan typ je een 1.
Mocht de afbeelding uit 2 of meer stukken lijken te bestaan, typ dan een 0.
De eerste afbeelding wordt nu geplot.
""")

for image_number in range(0, 200):
    # Construct image path
    image_location = os.path.join(this_folder, 'data')
    image_location = os.path.join(image_location, 'images')
    image_location = os.path.join(image_location, f'bag{bag_number}histogram_{image_number}.png')

    # Open in external window
    # Image.open(image_location).resize(size=(100,150)).show()

    # Plot image
    plt.imshow(np.asarray(Image.open(image_location)))
    plt.ion()
    plt.show()

    # Get label info
    label = input(f" 1 or 0? ")
    if not (label == "1" or label == "0"):
        print("Je moet een 1 of een 0 typen sukkel!")
        while True:
            label = input(f" 1 or 0? ")
            if label == "1" or label == "0": break

    csv_input = f"{bag_number}, {image_number}, {label}\n"
    file.write(csv_input)
print("""
Je hebt 200 afbeeldingen gelabeld, wees trots op jezelf en pak een stukje taart. 
Bedankt voor deze wetenschappelijke bijdrage aan nutteloosheid, euhm [BELANGRIJK PROJECT]
""")
file.close()
