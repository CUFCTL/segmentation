color_maps = {
    'cityscapes': [
        [128, 64,128], # road
        [244, 35,232], # sidewalk
        [ 70, 70, 70], # building
        [102,102,156], # wall
        [190,153,153], # fence
        [153,153,153], # pole
        [250,170, 30], # traffic light
        [220,220,  0], # traffic sign
        [107,142, 35], # vegetation
        [152,251,152], # terrain
        [ 70,130,180], # sky
        [220, 20, 60], # person
        [255,  0,  0], # rider
        [  0,  0,142], # car
        [  0,  0, 70], # truck
        [  0, 60,100], # bus
        [  0, 80,100], # train
        [  0,  0,230], # motorcycle
        [119, 11, 32], # bicyle 
        #[  0,  0,  0], # void
    ],
    'rellis': [
        [0, 0, 0], # void
        #[108, 64, 20], # dirt -> rainy day so dirt pixels were considered mud
        [0, 102, 0], # grass
        [0, 255, 0], # tree
        [0, 153, 153], # pole
        [0, 128, 255], # water
        [0, 0, 255], # sky
        [255, 255, 0], # vehicle
        [255, 0, 127], # object
        [64, 64, 64], # asphalt
        [255, 0, 0], # building
        [102, 0, 0], # log
        [204, 153, 255], # person
        [102, 0, 204], # fence
        [255, 153, 204], # bush
        [170, 170, 170], # concrete
        [41, 121, 255], # barrier
        [134, 255, 239], # puddle
        [99, 66, 34], # mud
        [110, 22, 138] # rubble
    ] 
}

labels = {
    'cityscapes': [
        "road",
        "buidling",
        "sidewalk",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicyle",
        "void"
    ],
    'rellis': [
        "void",
        "grass",
        "tree",
        "pole",
        "water",
        "sky",
        "vehicle",
        "object",
        "asphalt",
        "building",
        "log",
        "person",
        "fence",
        "bush",
        "concrete",
        "barrier",
        "puddle",
        "mud",
        "rubble"
    ] 
}