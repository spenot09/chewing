# Chewing
A Deep Learning inference repo built with tensorflow. 

## Prerequisites
  1. Install tensorflow (minimum 1.3)
  2. Install OpenCV
  3. All videos to run inference on must be in a single folder
  4. Create an output folder where you want the output to go
  
## Usage
Once you have cloned the repo to your desired directory, cd into it:

```
cd <DIRECTORY OF REPO>
```
Download the model file from: http://personal.ee.surrey.ac.uk/Personal/S.Hadfield/SCollins/retrained_graph.pb and copy it into  the tf_files folder.  

Then simply start the inference with the following command:

```
python -m scripts.inference  --videos=<FULL FILEPATH TO FOLDER WITH VIDEOS>  --output=<FILEPATH TO OUTPUT FOLDER>
```

And that's it. Eating rates should be outputted to the target folder.

## Output
The output folder should contain a single text file per video containg the mouthful rate of that video and a separate "main.txt" should be created which contains all mouthful rates of all the videos inferred.  
