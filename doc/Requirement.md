Proc_vision node

The Proc_vision node is a node use to generate the detection with the YOLO model and get all necessary information from it.

Requirements

Requirement 1

Titre : processAiActivationRequest
PreCondition : message to activate a camera and a model was received.
Post condition :
- the camera is active.
- the model is select.
- the name of model is send.

Requirement 2

Titre : imgDetection
PreCondition : 
- message containing a image was received.
- a depth image was received.
- a model was selected.
- a camera was selected.
Post condition : 
- data is set like distance and orientation.
- the object detection is transmitted.
- the depth is actualized.

Functional Exigence
Exigence 1

The node verify at the startup if cuda is ok.

Exigence 2

The range is configured to be between 0-25 meters.

Exigence 3

After a 2 second timeout, if there is no updated information from a node, it is initialized.

Exigence 4
Check periodically if the msg recieved has consistent publish stamp with a 20% tolorence