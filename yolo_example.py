from ultralytics import YOLO

model = YOLO("yolo26n.pt")  

#results = model("data/bus.jpg")

#build a yaml file for your own data set and train the model on your own data set
#minimal_circle_detect.py change learning rate and epochs and batch size and image size and optimizer and loss function and early stopping criteria
#clss is an where it is at coresponding text and image files ``Z
#at least 2 classes 
#how many classes names for classes two folers and one folder is for images and the other is for the labels
#set patience criteria for early stopping and save the best model

# predict level results list 
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()
    result.save(filename ="results/bus.jpg")

#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

d
#collect data and train modelos on your own data sets and detect ur own examples
    