import os
def names():
    x = os.listdir("Training-data")
    y = [""]
    for folder in x:
        z = "Training-data" + "/" + folder
        w = os.listdir(z)
        for i in w:
            if i.startswith("_"):
                p = i.replace("_","")
                y.append(p)
    return y
