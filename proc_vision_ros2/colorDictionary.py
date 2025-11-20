import yaml

colorDictionary = {}

try:
    with open('/home/sonia/ros2_sonia_ws/src/proc_vision_ros2/models/sim-2025/data.yaml', 'r') as file:
        data = yaml.safe_load(file)
        nomDetectionDic = data["names"]
        nbDectection = data["nc"]
        rgbIncrement = 255//(nbDectection//3)
        red = 0
        green = 0
        blue = 0
        counter = 0
        for nameDetection in nomDetectionDic.values():
            colorDictionary[nameDetection] = (red,blue,green)   

            if(counter % 3 == 0):
                red = red + rgbIncrement
            elif(counter % 3 == 1):
                green = green + rgbIncrement
            else:
                blue = blue + rgbIncrement

            counter += 1

    print(colorDictionary)
except FileNotFoundError:
    print("Error: 'config.yaml' not found.")
except yaml.YAMLError as exc:
    print(f"Error parsing YAML file: {exc}")




# Color dictionary.
# Def: each color is a RBG code (R,G,B)
